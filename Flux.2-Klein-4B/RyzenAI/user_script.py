# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Olive user_script for FLUX.2-klein-4B ONNX export.

import torch
import torch.nn as nn
from torch.onnx import symbolic_helper


try:
    from torch.onnx._type_utils import JitScalarType
except (ImportError, ModuleNotFoundError):
    from torch.onnx import JitScalarType


# =============================================================================
# Transformer
# =============================================================================

# ---------------------------------------------------------------------------
# RMSNorm decomposition for opset 17
#
# aten::rms_norm has no native ONNX op at opset 17. Decompose it as:
#   Cast → Pow(2) → ReduceMean(axes_i, keepdims=1) → Add(eps) → Sqrt
#        → Div → Cast → Mul(weight)
#
# keepdims=1 preserves the reduced dim as size-1, so Div broadcasts without
# an extra Unsqueeze. The resulting 8-node subgraph is fused into
# SimplifiedLayerNormalization by the onnxruntime.transformers mmdit optimizer.
# ---------------------------------------------------------------------------

@symbolic_helper.parse_args("v", "is", "v", "v")
def _rms_norm_symbolic(g, input, normalized_shape, weight, eps):
    eps_val = symbolic_helper._maybe_get_const(eps, "f")
    if eps_val is None or not isinstance(eps_val, (int, float)):
        eps_val = 1e-6

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    input_dtype = JitScalarType.from_value(input, JitScalarType.FLOAT)
    fp32_onnx   = JitScalarType.FLOAT.onnx_type()

    input_fp32     = g.op("Cast",     input,     to_i=fp32_onnx)
    pow_two        = g.op("Constant", value_t=torch.tensor(2.0, dtype=torch.float32))
    x_squared      = g.op("Pow",      input_fp32, pow_two)
    x_squared_mean = g.op("ReduceMean", x_squared, axes_i=axes)
    eps_const      = g.op("Constant", value_t=torch.tensor(eps_val, dtype=torch.float32))
    rms            = g.op("Sqrt", g.op("Add", x_squared_mean, eps_const))
    normalized     = g.op("Cast", g.op("Div", input_fp32, rms), to_i=input_dtype.onnx_type())

    if weight is not None and not symbolic_helper._is_none(weight):
        normalized = g.op("Mul", normalized, weight)

    normalized.setType(input.type())
    return normalized


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class FluxTransformerWrapper(nn.Module):
    """Flux2Transformer2DModel wrapper for ONNX export.

    guidance is passed as None internally — not an ONNX input.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _merge_lora_into_transformer(transformer: nn.Module, lora_ckpt_path: str) -> None:
    """Load LoRA weights from te_swap_adapter.pt and merge them (W += B @ A * scale)
    into the matching Linear modules of the transformer in-place.

    Matching targets: .attn.(to_k|to_v|add_k_proj|add_v_proj)
    The checkpoint stores lora tensors as [A0, B0, A1, B1, ...] in module-discovery
    order, which is the exact order produced by inject_lora() in lora.py.
    """
    import re
    import sys as _sys
    from pathlib import Path as _Path

    sd = torch.load(lora_ckpt_path, map_location="cpu")
    lora_tensors = sd.get("lora")
    if not lora_tensors:
        print(f"  [LORA] No 'lora' key in checkpoint {lora_ckpt_path}; skipping LoRA merge.")
        return

    cfg = sd.get("config", {"lora_rank": 64})
    rank  = cfg.get("lora_rank", 64)
    scale = 1.0  # alpha == rank, so scale = alpha/rank = 1.0

    pattern = re.compile(r"\.attn\.(to_k|to_v|add_k_proj|add_v_proj)$")
    matched_linears: list[nn.Linear] = []
    for name, mod in transformer.named_modules():
        for child_name, child in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and pattern.search(full):
                matched_linears.append(child)

    expected_pairs = len(matched_linears)
    if len(lora_tensors) != expected_pairs * 2:
        print(
            f"  [LORA] LoRA tensor count mismatch: "
            f"checkpoint has {len(lora_tensors)}, "
            f"model has {expected_pairs} target linears × 2. Skipping LoRA merge."
        )
        return

    merged = 0
    for i, linear in enumerate(matched_linears):
        A = lora_tensors[i * 2].to(dtype=linear.weight.dtype, device=linear.weight.device)   # [rank, in]
        B = lora_tensors[i * 2 + 1].to(dtype=linear.weight.dtype, device=linear.weight.device)  # [out, rank]
        linear.weight.data.add_(B @ A, alpha=scale)
        merged += 1

    print(f"  [LORA] Merged LoRA into {merged} Linear modules (rank={rank}, scale={scale:.3f}).")


def transformer_load(model_path: str, lora_ckpt: str | None = None) -> FluxTransformerWrapper:
    """Load Flux2Transformer2DModel and optionally merge LoRA weights.

    lora_ckpt can be passed directly or via the OLIVE_LORA_CKPT environment
    variable (set by export_models.py when --lora_ckpt is provided), because
    Olive's model_loader mechanism only passes model_path.
    """
    import os as _os
    from diffusers import Flux2Transformer2DModel

    # Register the RMSNorm custom symbolic here so it only affects the
    # transformer export, not other components (text encoder, VAE).
    torch.onnx.register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, 17)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    transformer.eval()
    transformer.to(device=device)

    from pathlib import Path as _Path
    _default_lora = _Path(__file__).parent / "weights" / "te_swap_adapter.pt"
    resolved_ckpt = lora_ckpt or _os.environ.get("OLIVE_LORA_CKPT") or (
        str(_default_lora) if _default_lora.exists() else None
    )
    if resolved_ckpt:
        print(f"  [LORA] Merging LoRA weights from {resolved_ckpt} into transformer ...")
        _merge_lora_into_transformer(transformer, resolved_ckpt)

    return FluxTransformerWrapper(transformer)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# Model config: in_channels=128, joint_attention_dim=7680,
#               axes_dims_rope=[32,32,32,32] → rope last dim = 4
# Resolution:   1024×1024 → img_seq_len = (1024/16)² = 4096
#
# img_ids / txt_ids are INT64 so Cast(INT64→FLOAT32) nodes are traced
# into the graph (the transformer converts position indices to float internally).
# ---------------------------------------------------------------------------

_BATCH       = 1
_IMG_SEQ_LEN = 4096
_TXT_SEQ_LEN = 256
_HIDDEN_DIM  = 128
_TXT_DIM     = 7680
_ROPE_DIMS   = 4


def transformer_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "hidden_states": torch.randn(
            _BATCH, _IMG_SEQ_LEN, _HIDDEN_DIM, dtype=torch.float32, device=device
        ),
        "encoder_hidden_states": torch.randn(
            _BATCH, _TXT_SEQ_LEN, _TXT_DIM, dtype=torch.float32, device=device
        ),
        "timestep": torch.tensor([0.5] * _BATCH, dtype=torch.float32, device=device),
        "img_ids": torch.zeros(
            _BATCH, _IMG_SEQ_LEN, _ROPE_DIMS, dtype=torch.int64, device=device
        ),
        "txt_ids": torch.zeros(
            _BATCH, _TXT_SEQ_LEN, _ROPE_DIMS, dtype=torch.int64, device=device
        ),
    }


# =============================================================================
# Text Encoder (Qwen3)
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class Qwen3TextEncoderWrapper(nn.Module):
    """Qwen3 text encoder wrapper for ONNX export.

    Stacks hidden states from the specified layers and reshapes them into
    prompt_embeds expected by the Flux2-Klein transformer
    (shape: [batch, seq_len, num_layers * hidden_dim]).

    Pre-computes the 4D additive float causal mask and passes it as a dict
    so that Qwen3Model.forward skips create_causal_mask entirely. This is
    required because create_causal_mask internally uses _vmap_for_bhqkv +
    .item(), which fails under ONNX JIT tracing (RuntimeError: invalid
    unordered_map key). The same issue also affects dynamo tracing.
    """

    def __init__(self, model: nn.Module, hidden_states_layers: tuple[int, ...] = (9, 18, 27)) -> None:
        super().__init__()
        self.model = model
        self.hidden_states_layers = hidden_states_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=input_ids.device)
        kv_arange = torch.arange(seq_len, device=input_ids.device)

        # Lower-triangular causal mask: kv <= q position → can attend
        bool_mask = kv_arange.unsqueeze(0) <= cache_position.unsqueeze(1)  # [q, kv]
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)                     # [1, 1, q, kv]
        if attention_mask is not None:
            bool_mask = bool_mask & attention_mask[:, None, None, :].bool() # apply padding mask

        dtype = next(self.model.parameters()).dtype
        float_mask = torch.where(
            bool_mask,
            torch.zeros(1, dtype=dtype, device=input_ids.device),
            torch.full((1,), torch.finfo(dtype).min, dtype=dtype, device=input_ids.device),
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": float_mask},
            position_ids=position_ids,
            cache_position=cache_position,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        stacked = torch.stack([outputs.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        batch_size, num_channels, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def text_encoder_load(model_path: str) -> Qwen3TextEncoderWrapper:
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_path, subfolder="text_encoder")
    config.use_cache = False
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        subfolder="text_encoder",
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    return Qwen3TextEncoderWrapper(model).eval()


# ---------------------------------------------------------------------------
# Dummy inputs
#
# Sequence length matches _TXT_SEQ_LEN used by the transformer.
# ---------------------------------------------------------------------------

_TEXT_BATCH   = 1
_TEXT_SEQ_LEN = 256


def text_encoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.zeros((_TEXT_BATCH, _TEXT_SEQ_LEN), dtype=torch.long, device=device)
    attention_mask = torch.ones((_TEXT_BATCH, _TEXT_SEQ_LEN), dtype=torch.long, device=device)
    position_ids = (
        torch.arange(_TEXT_SEQ_LEN, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(_TEXT_BATCH, -1)
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


# =============================================================================
# VAE
# =============================================================================

try:
    from diffusers import AutoencoderKLFlux2  # type: ignore
except Exception:
    AutoencoderKLFlux2 = None


# =============================================================================
# VAE Encoder
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class VaeEncoderWrapper(nn.Module):
    """AutoencoderKL encoder wrapper for ONNX export.

    Uses latent_dist.mode() for deterministic, traceable output.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.mode()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def vae_encoder_load(model_path: str) -> VaeEncoderWrapper:
    from diffusers import AutoencoderKL

    vae_cls = AutoencoderKLFlux2 if AutoencoderKLFlux2 is not None else AutoencoderKL
    print(f"[INFO] using VAE class: {vae_cls.__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae_cls.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    vae.to(device=device)
    return VaeEncoderWrapper(vae)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# 1024×1024 RGB image input
# ---------------------------------------------------------------------------

_VAE_ENC_BATCH = 1
_VAE_ENC_C     = 3      # RGB
_VAE_ENC_H     = 1024
_VAE_ENC_W     = 1024


def vae_encoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "images": torch.randn(
            _VAE_ENC_BATCH, _VAE_ENC_C, _VAE_ENC_H, _VAE_ENC_W,
            dtype=torch.float32,
            device=device,
        )
    }


# =============================================================================
# VAE Decoder
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class VaeDecoderWrapper(nn.Module):
    """AutoencoderKL decoder wrapper for ONNX export."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def vae_decoder_load(model_path: str) -> VaeDecoderWrapper:
    from diffusers import AutoencoderKL

    vae_cls = AutoencoderKLFlux2 if AutoencoderKLFlux2 is not None else AutoencoderKL
    print(f"[INFO] using VAE class: {vae_cls.__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae_cls.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    vae.to(device=device)
    return VaeDecoderWrapper(vae)


# ---------------------------------------------------------------------------
# Dummy inputs
#
# 1024×1024 output → latent spatial 128×128 (VAE 8× factor), 32 channels
# 32ch × 2×2 spatial pack = 128ch → transformer in_channels=128
# ---------------------------------------------------------------------------

_VAE_BATCH    = 1
_VAE_LATENT_C = 32   # AutoencoderKLFlux2 latent channels
_VAE_LATENT_H = 128
_VAE_LATENT_W = 128


def vae_decoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "latents": torch.randn(
            _VAE_BATCH, _VAE_LATENT_C, _VAE_LATENT_H, _VAE_LATENT_W,
            dtype=torch.float32,
            device=device,
        )
    }


# =============================================================================
# Gemma-4-E2B Text Encoder (TE-swap adapter source encoder)
# =============================================================================

# ---------------------------------------------------------------------------
# Wrapper
#
# Gemma-4-E2B is a multimodal model; only the text tower (language_model,
# 35 layers, hidden 1536) is used.  We extract four hidden states at layers
# {9, 18, 26, 30} — the candidate-layer superset consumed by
# GemmaToKleinAdapterV2 — and return them as individual fp32 tensors so the
# adapter can apply its per-tap var-norm + learnable softmax mixing.
#
# The attention_mask is passed as a plain [B, seq] int64 tensor.  Gemma4's
# text tower accepts this natively when attn_implementation="eager", which
# also avoids the scaled_dot_product_attention / .item() trace failures that
# affect the causal-mask path.
# ---------------------------------------------------------------------------

_GEMMA_LAYER_PICKS = (9, 18, 26, 30)


class GemmaTextEncoderWrapper(nn.Module):
    """Gemma-4-E2B text tower wrapper for ONNX export.

    Returns four hidden-state taps (layers 9, 18, 26, 30), each
    [B, seq, 1536], as separate fp32 tensors named tap_9/18/26/30.
    output_hidden_states is hardcoded True so torch.jit.trace never
    sees it as a dynamic branch.
    """

    def __init__(self, text_model: nn.Module, layer_picks: tuple = _GEMMA_LAYER_PICKS) -> None:
        super().__init__()
        self.text_model = text_model
        self.layer_picks = layer_picks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = outputs.hidden_states  # tuple len = n_layers + 1
        return tuple(hs[i] for i in self.layer_picks)  # 4 x [B, seq, 1536]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def gemma_text_encoder_load(model_path: str) -> GemmaTextEncoderWrapper:
    """Load Gemma-4-E2B, extract the text tower, discard vision components."""
    import gc
    try:
        from transformers import Gemma4ForConditionalGeneration
    except ImportError:
        import transformers
        raise ImportError(
            f"Gemma4ForConditionalGeneration is not available in the installed "
            f"transformers {transformers.__version__}. "
            "Upgrade to transformers >= 5.0: "
            "pip install 'transformers>=5.0.0'"
        ) from None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    # Extract the pure text backbone (Gemma4TextModel).
    lang = getattr(full, "language_model", None)
    if lang is None:
        lang = getattr(getattr(full, "model", full), "language_model", None)
    if lang is None or not hasattr(lang, "layers"):
        raise RuntimeError(
            "Could not locate Gemma4TextModel (.language_model) in the loaded model"
        )

    # Detach and free the multimodal components we don't need.
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector", "lm_head"):
        if hasattr(full, attr):
            setattr(full, attr, None)
    del full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for p in lang.parameters():
        p.requires_grad_(False)

    lang.to(device=device).eval()
    return GemmaTextEncoderWrapper(lang).eval()


# ---------------------------------------------------------------------------
# Dummy inputs
#
# seq_len=512 matches GemmaTextEncoder.max_sequence_length in
# te_swap_adapter/src/gemma_text_encoder.py.
# ---------------------------------------------------------------------------

_GEMMA_BATCH   = 1
_GEMMA_SEQ_LEN = 512


def gemma_text_encoder_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "input_ids":      torch.zeros((_GEMMA_BATCH, _GEMMA_SEQ_LEN), dtype=torch.long, device=device),
        "attention_mask": torch.ones((_GEMMA_BATCH, _GEMMA_SEQ_LEN),  dtype=torch.long, device=device),
    }


# =============================================================================
# Gemma-4-E2B + GemmaToKleinAdapterV2 — Combined ONNX wrapper
# =============================================================================

from adapter import GemmaToKleinAdapterV2  # local adapter.py (no external dependency)

# ---------------------------------------------------------------------------
# Combined wrapper
#
# Exposes the same interface as the original Qwen text_encoder:
#   forward(input_ids, attention_mask) -> prompt_embeds [B, seq, 7680]
#
# Internally it runs the Gemma text tower to extract the 4 hidden-state taps
# (layers 9/18/26/30) and passes them through GemmaToKleinAdapterV2.
# The attention_mask (int64 right-pad) is passed to both the Gemma model and
# the adapter's BidirectionalRefiner.  position_ids are NOT required.
# ---------------------------------------------------------------------------

_ADAPTER_GEMMA_HIDDEN = 1536
_ADAPTER_KLEIN_DIM    = 7680
_ADAPTER_CANDIDATES   = (9, 18, 26, 30)


class GemmaWithAdapterWrapper(nn.Module):
    """Gemma-4-E2B text tower + GemmaToKleinAdapterV2, fused into one nn.Module.

    Input:  input_ids [B, seq], attention_mask [B, seq]
    Output: prompt_embeds [B, seq, 7680]

    This is a drop-in replacement for the Qwen3 text_encoder ONNX
    (no position_ids, seq=512 instead of 256).
    """

    def __init__(self, gemma_text_model: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.gemma = gemma_text_model   # Gemma4TextModel (text tower only)
        self.adapter = adapter          # GemmaToKleinAdapterV2

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, seq]  int64
        attention_mask: torch.Tensor,   # [B, seq]  int64
    ) -> torch.Tensor:                  # [B, seq, 7680]
        out = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states          # tuple, index 0 = embedding, 1..N = layers
        taps = [hs[i] for i in _ADAPTER_CANDIDATES]
        return self.adapter(taps, attention_mask=attention_mask.bool())


# ---------------------------------------------------------------------------
# Model loader
#
# model_path  →  Gemma-4-E2B HF repo ID or local path
# OLIVE_ADAPTER_CKPT env var  →  path to te_swap_adapter.pt
#   (set by export_models.py via --adapter_ckpt, or hard-set in the
#    environment before running olive run manually)
# ---------------------------------------------------------------------------

def gemma_with_adapter_load(model_path: str) -> GemmaWithAdapterWrapper:
    """Load Gemma-4-E2B text tower + GemmaToKleinAdapterV2 as a single module.

    Both sub-models are kept in fp32 for stable ONNX tracing; the downstream
    OrtTransformersOptimization fp16 pass converts weights after export.
    """
    import gc as _gc
    import os as _os
    from pathlib import Path as _Path

    # Register aten::rms_norm → custom decomposed symbolic so torch.onnx.export
    # at opset 17 can handle both Gemma's RMSNorm and the adapter's nn.RMSNorm.
    torch.onnx.register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, 17)

    # ---- resolve adapter checkpoint (env var → default bundled path) ----
    _default_ckpt = _Path(__file__).parent / "weights" / "te_swap_adapter.pt"
    adapter_ckpt  = _Path(_os.environ.get("OLIVE_ADAPTER_CKPT", str(_default_ckpt)))
    if not adapter_ckpt.exists():
        raise RuntimeError(
            f"Adapter checkpoint not found: {adapter_ckpt}\n"
            "Place te_swap_adapter.pt in the weights/ directory, or set "
            "OLIVE_ADAPTER_CKPT to the correct path."
        )

    # ---- load Gemma text tower ----
    try:
        from transformers import Gemma4ForConditionalGeneration
    except ImportError:
        import transformers
        raise ImportError(
            f"Gemma4ForConditionalGeneration is not available in the installed "
            f"transformers {transformers.__version__}. "
            "Upgrade: pip install 'transformers>=5.0.0'"
        ) from None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    lang = getattr(full, "language_model", None)
    if lang is None:
        lang = getattr(getattr(full, "model", full), "language_model", None)
    if lang is None or not hasattr(lang, "layers"):
        raise RuntimeError("Could not locate Gemma4TextModel (.language_model)")

    for attr in ("vision_tower", "audio_tower", "multi_modal_projector", "lm_head"):
        if hasattr(full, attr):
            setattr(full, attr, None)
    del full
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for p in lang.parameters():
        p.requires_grad_(False)
    lang.to(device=device).eval()

    # ---- load adapter ----
    sd = torch.load(str(adapter_ckpt), map_location="cpu")
    cfg = sd.get("config", {"refiner_depth": 2})
    adapter = GemmaToKleinAdapterV2(
        candidates=_ADAPTER_CANDIDATES,
        refiner_depth=cfg.get("refiner_depth", 2),
    )
    ema_state = sd.get("ema", sd.get("adapter", sd))
    adapter.load_state_dict(ema_state, strict=False)
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.to(device=device, dtype=torch.float32).eval()

    return GemmaWithAdapterWrapper(lang, adapter).eval()


# ---------------------------------------------------------------------------
# Dummy inputs
# ---------------------------------------------------------------------------

_GEMMA_ADAPTER_BATCH   = 1
_GEMMA_ADAPTER_SEQ_LEN = 512


def gemma_with_adapter_conversion_inputs(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "input_ids":      torch.zeros(
            (_GEMMA_ADAPTER_BATCH, _GEMMA_ADAPTER_SEQ_LEN), dtype=torch.long, device=device
        ),
        "attention_mask": torch.ones(
            (_GEMMA_ADAPTER_BATCH, _GEMMA_ADAPTER_SEQ_LEN), dtype=torch.long, device=device
        ),
    }
