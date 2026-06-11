# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Export Qwen3 prompt_embeds ONNX compatible with Flux.2-klein runtime:
#   1. torch.onnx.export (fp16, eager attention, position_ids input)
#   2. MatMul -> com.microsoft MatMulNBits (block_size=128)
#
# Matches flux.2-klein/flux.2-klein-4B-onnx/text_encoder/export_qwen3_prompt_embeds_matmulnbits.py

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

MS_DOMAIN = "com.microsoft"
PROMPT_HIDDEN_STATE_LAYERS = (9, 18, 27)


class Qwen3PromptEmbedWrapper(nn.Module):
    """Stacks prompt_embeds layers; pre-computes causal mask to avoid create_causal_mask vmap."""

    def __init__(
        self,
        model: nn.Module,
        hidden_states_layers: tuple[int, ...] = PROMPT_HIDDEN_STATE_LAYERS,
    ) -> None:
        super().__init__()
        self.model = model
        self.hidden_states_layers = hidden_states_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=input_ids.device)
        kv_arange = torch.arange(seq_len, device=input_ids.device)

        bool_mask = kv_arange.unsqueeze(0) <= cache_position.unsqueeze(1)
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            bool_mask = bool_mask & attention_mask[:, None, None, :].bool()

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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def _make_dummy_inputs(
    config,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_id = getattr(config, "bos_token_id", None)
    if token_id is None or token_id >= config.vocab_size:
        token_id = min(1, config.vocab_size - 1)
    input_ids = torch.full((batch_size, sequence_length), token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=device)
    position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return input_ids, attention_mask, position_ids


def _export_fp16_onnx(
    model_dir: Path,
    output_path: Path,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    hidden_states_layers: tuple[int, ...],
    opset: int = 17,
) -> None:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.use_cache = False
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)

    wrapper = Qwen3PromptEmbedWrapper(model, hidden_states_layers).eval()
    input_ids, attention_mask, position_ids = _make_dummy_inputs(config, batch_size, sequence_length, device)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "prompt_embeds": {0: "batch_size", 1: "sequence_length"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, position_ids),
            str(output_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["prompt_embeds"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            export_params=True,
            do_constant_folding=True,
        )

    del model, wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _quantize_matmul_weight_kn(weight: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight_b = np.ascontiguousarray(weight.astype(np.float16))
    rows, cols = weight_b.shape
    kpack = 2
    k_blocks = math.ceil(rows / block_size)
    blob_size = (block_size + kpack - 1) // kpack
    padded_rows = k_blocks * block_size
    if padded_rows != rows:
        weight_b = np.pad(weight_b, ((0, padded_rows - rows), (0, 0)), "constant")

    packed = np.zeros((cols, k_blocks, blob_size), dtype=np.uint8)
    scales = np.zeros((cols, k_blocks), dtype=np.float16)
    zero_points = np.zeros((cols, math.ceil(k_blocks / kpack)), dtype=np.uint8)
    quantize_matmul_4bits(packed, weight_b, scales, zero_points, block_size, cols, rows, False)
    return packed, scales.reshape(-1), zero_points.reshape(-1)


def _ensure_ms_opset(model: onnx.ModelProto) -> None:
    if not any(op.domain == MS_DOMAIN for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid(MS_DOMAIN, 1))


def _convert_matmul_to_matmul_nbits(model: onnx.ModelProto, *, block_size: int, bits: int) -> tuple[int, int]:
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    init_by_name = {init.name: init for init in graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    remove_names: set[str] = set()
    converted = 0
    skipped = 0

    for node in graph.node:
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue

        weight_name = next((name for name in node.input if name in initializer_names), None)
        if weight_name is None:
            new_nodes.append(node)
            skipped += 1
            continue

        weight = onnx.numpy_helper.to_array(init_by_name[weight_name])
        if weight.ndim != 2:
            new_nodes.append(node)
            skipped += 1
            continue

        k_dim, n_dim = int(weight.shape[0]), int(weight.shape[1])
        packed, scales, zero_points = _quantize_matmul_weight_kn(weight, block_size)

        qweight_name = f"{weight_name}_MatMulNBits_qweight"
        scales_name = f"{weight_name}_MatMulNBits_scales"
        qzeros_name = f"{weight_name}_MatMulNBits_qzeros"
        new_initializers.extend(
            [
                onnx.numpy_helper.from_array(packed, name=qweight_name),
                onnx.numpy_helper.from_array(scales, name=scales_name),
                onnx.numpy_helper.from_array(zero_points, name=qzeros_name),
            ]
        )

        activation_input = node.input[0] if node.input[0] != weight_name else node.input[1]
        new_nodes.append(
            helper.make_node(
                "MatMulNBits",
                inputs=[activation_input, qweight_name, scales_name, qzeros_name],
                outputs=list(node.output),
                name=node.name or f"{weight_name}_MatMulNBits",
                domain=MS_DOMAIN,
                K=k_dim,
                N=n_dim,
                bits=bits,
                block_size=block_size,
                accuracy_level=0,
            )
        )
        remove_names.add(weight_name)
        converted += 1

    if converted:
        kept_initializers = [init for init in graph.initializer if init.name not in remove_names]
        del graph.initializer[:]
        graph.initializer.extend(kept_initializers)
        graph.initializer.extend(new_initializers)
        del graph.node[:]
        graph.node.extend(new_nodes)

    return converted, skipped


def _save_model_external(model: onnx.ModelProto, output_path: Path) -> None:
    data_file = output_path.name + ".data"
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
    )


def quantize_fp16_prompt_embeds_onnx_to_matmulnbits(
    fp16_onnx_path: str | Path,
    output_path: str | Path,
    *,
    block_size: int = 128,
    bits: int = 4,
) -> tuple[int, int]:
    """Load an fp16 prompt_embeds ONNX (e.g. from Olive ModelBuilder) and write MatMulNBits INT4 ONNX.

    Only ``MatMul`` nodes with a constant 2-D weight initializer are converted (same rules as
    :func:`export_prompt_embeds_matmulnbits` torch-export path). Graphs that use only ``Gemm`` or
    fused ops may report ``converted=0``; use the built-in torch export path in that case.

    Returns:
        ``(converted_count, skipped_matmul_count)`` from :func:`_convert_matmul_to_matmul_nbits`.
    """
    fp16_onnx_path = Path(fp16_onnx_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MATMULNBITS] fp16 ONNX : {fp16_onnx_path}")
    print(f"[MATMULNBITS] output    : {output_path}")

    model = onnx.load(str(fp16_onnx_path), load_external_data=True)
    _ensure_ms_opset(model)
    converted, skipped = _convert_matmul_to_matmul_nbits(model, block_size=block_size, bits=bits)
    print(f"[MATMULNBITS] MatMul converted: {converted}, skipped: {skipped}")
    _save_model_external(model, output_path)

    if output_path.exists():
        data_path = output_path.parent / (output_path.name + ".data")
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"[MATMULNBITS] Wrote {output_path} ({size_mb:.1f} MiB)")
        if data_path.exists():
            print(f"[MATMULNBITS] Wrote {data_path} ({data_path.stat().st_size / (1024 ** 3):.2f} GiB)")

    return converted, skipped


def export_prompt_embeds_matmulnbits(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 1,
    sequence_length: int = 128,
    device: str = "auto",
    hidden_states_layers: tuple[int, ...] = PROMPT_HIDDEN_STATE_LAYERS,
    block_size: int = 128,
    bits: int = 4,
    opset: int = 17,
    fp16_onnx_path: str | Path | None = None,
) -> Path:
    """Export Flux-compatible prompt_embeds MatMulNBits ONNX.

    If ``fp16_onnx_path`` is set, skips PyTorch ONNX export and quantizes that graph instead
    (e.g. after Olive ModelBuilder fp16 + ``hidden_states_layers``).
    """
    model_dir = Path(model_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_layers = tuple(hidden_states_layers)

    print(f"[MATMULNBITS] model_dir : {model_dir}")
    print(f"[MATMULNBITS] output    : {output_path}")
    print(f"[MATMULNBITS] layers    : {hidden_layers}")

    if fp16_onnx_path is not None:
        print(f"[MATMULNBITS] source    : fp16 ONNX ({fp16_onnx_path})")
        quantize_fp16_prompt_embeds_onnx_to_matmulnbits(
            fp16_onnx_path,
            output_path,
            block_size=block_size,
            bits=bits,
        )
        return output_path

    resolved_device = _resolve_device(device)
    print(f"[MATMULNBITS] device    : {resolved_device}")

    with tempfile.TemporaryDirectory(prefix="qwen3_prompt_embeds_") as tmp:
        fp16_onnx = Path(tmp) / "prompt_embeds_fp16.onnx"
        _export_fp16_onnx(
            model_dir,
            fp16_onnx,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=resolved_device,
            hidden_states_layers=hidden_layers,
            opset=opset,
        )

        quantize_fp16_prompt_embeds_onnx_to_matmulnbits(
            fp16_onnx,
            output_path,
            block_size=block_size,
            bits=bits,
        )

    return output_path
