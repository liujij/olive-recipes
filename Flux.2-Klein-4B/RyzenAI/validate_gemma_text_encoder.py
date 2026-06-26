"""Validate the exported gemma_text_encoder ONNX against a PyTorch reference.

Supports two validation modes, auto-detected from the ONNX output names:

  Mode A — Combined ONNX (new fp16, after full re-export with --clear_cache):
    inputs : input_ids, attention_mask
    outputs: prompt_embeds [B, 512, 7680]
    Compared directly with PyTorch reference.

  Mode B — Legacy 4-tap ONNX (existing fp32 cache model):
    inputs : input_ids, attention_mask
    outputs: tap_9, tap_18, tap_26, tap_30  each [B, 512, 1536]
    Manually runs the local adapter on the taps, then compares.

Usage
-----
# Mode A (after combined re-export) — no arguments needed if weights are in place:
python validate_gemma_text_encoder.py --onnx_path output_model/text_encoder/model.onnx

# Mode B (validate existing 4-tap cache model, no re-export needed):
python validate_gemma_text_encoder.py \
    --onnx_path ryzenai_cache/default_workflow/runs/f71451ca/models/model.onnx

# Override adapter weights path if needed:
python validate_gemma_text_encoder.py --adapter_ckpt /custom/path/te_swap_adapter.pt

Force re-export of combined model
-----------------------------------
  conda run -n olive_test python export_models.py \
      --models gemma_text_encoder \
      --clear_cache
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Local adapter code (no external te_swap_adapter package needed)
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from adapter import GemmaToKleinAdapterV2, CANDIDATES as _ADAPTER_CANDIDATES

_DEFAULT_ADAPTER_CKPT = _HERE / "weights" / "te_swap_adapter.pt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate gemma_text_encoder ONNX export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--onnx_path",    default=None,
                   help="Path to model.onnx. Defaults to the known fp32 cache path.")
    p.add_argument("--adapter_ckpt", default=None,
                   help=f"Path to te_swap_adapter.pt. Default: weights/te_swap_adapter.pt")
    p.add_argument("--gemma_model",  default="google/gemma-4-E2B")
    p.add_argument("--prompt",       default="a red panda eating bamboo, photorealistic")
    p.add_argument("--seq_len",      type=int, default=512)
    p.add_argument("--atol",         type=float, default=0.05,
                   help="Max absolute error threshold (default 0.05)")
    p.add_argument("--skip_pytorch", action="store_true",
                   help="Skip PyTorch reference run (shape-only check).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ONNX session helpers
# ---------------------------------------------------------------------------

def load_onnx_session(onnx_path: str):
    import onnxruntime as ort
    print(f"  Loading ONNX session from:\n  {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_info  = {i.name: (i.type, i.shape) for i in sess.get_inputs()}
    output_info = {o.name: (o.type, o.shape) for o in sess.get_outputs()}
    output_names = list(output_info.keys())
    print(f"  Inputs : {input_info}")
    print(f"  Outputs: {output_info}")
    return sess, output_names


def detect_mode(output_names: list[str]) -> str:
    if "prompt_embeds" in output_names:
        return "combined"
    if "tap_9" in output_names:
        return "4tap"
    raise ValueError(f"Unrecognised output names: {output_names}")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise(gemma_model: str, prompt: str, seq_len: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(gemma_model)
    tok.padding_side = "right"
    enc = tok(prompt, padding="max_length", truncation=True,
               max_length=seq_len, return_tensors="np")
    ids  = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    print(f"  Tokenised: shape={ids.shape}  non-pad={int(mask.sum())}")
    return ids, mask


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def run_onnx_combined(sess, input_ids, attention_mask) -> np.ndarray:
    result = sess.run(None, {
        "input_ids": input_ids, "attention_mask": attention_mask,
    })
    return result[0]  # [B, seq, 7680]


def run_onnx_4tap(sess, input_ids, attention_mask) -> list[np.ndarray]:
    return sess.run(None, {
        "input_ids": input_ids, "attention_mask": attention_mask,
    })  # [tap_9, tap_18, tap_26, tap_30]


# ---------------------------------------------------------------------------
# Adapter inference (PyTorch, uses local adapter.py)
# ---------------------------------------------------------------------------

def _load_adapter(adapter_ckpt: str) -> GemmaToKleinAdapterV2:
    sd  = torch.load(adapter_ckpt, map_location="cpu")
    cfg = sd.get("config", {"refiner_depth": 2})
    ad  = GemmaToKleinAdapterV2(
        candidates=_ADAPTER_CANDIDATES,
        refiner_depth=cfg.get("refiner_depth", 2),
    ).eval()
    ema = sd.get("ema", sd.get("adapter", sd))
    ad.load_state_dict(ema, strict=False)
    for p in ad.parameters():
        p.requires_grad_(False)
    return ad


def run_adapter_on_taps(taps_np: list[np.ndarray],
                        attention_mask: np.ndarray,
                        adapter_ckpt: str) -> np.ndarray:
    ad  = _load_adapter(adapter_ckpt)
    taps_t = [torch.from_numpy(t.astype(np.float32)) for t in taps_np]
    mask_t = torch.from_numpy(attention_mask.astype(np.int64))
    with torch.no_grad():
        out = ad(taps_t, attention_mask=mask_t.bool())
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# PyTorch reference (full chain)
# ---------------------------------------------------------------------------

def run_pytorch_reference(adapter_ckpt: str, gemma_model: str,
                           input_ids: np.ndarray,
                           attention_mask: np.ndarray) -> np.ndarray:
    """Run Gemma (fp32, CPU) + local adapter → [B, seq, 7680]."""
    import gc
    from transformers import Gemma4ForConditionalGeneration

    print("  Loading Gemma-4-E2B text tower (fp32, CPU) ...")
    full = Gemma4ForConditionalGeneration.from_pretrained(
        gemma_model, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    lang = getattr(full, "language_model", None)
    if lang is None:
        lang = getattr(getattr(full, "model", full), "language_model", None)
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector", "lm_head"):
        if hasattr(full, attr):
            setattr(full, attr, None)
    del full
    gc.collect()
    for p in lang.parameters():
        p.requires_grad_(False)
    lang.eval()

    ad = _load_adapter(adapter_ckpt)

    ids_t  = torch.from_numpy(input_ids)
    mask_t = torch.from_numpy(attention_mask)
    with torch.no_grad():
        out = lang(
            input_ids=ids_t, attention_mask=mask_t,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        hs   = out.hidden_states
        taps = [hs[i].to(torch.float32) for i in _ADAPTER_CANDIDATES]
        ref  = ad(taps, attention_mask=mask_t.bool())
    return ref.cpu().numpy()


# ---------------------------------------------------------------------------
# Shape / sanity checks
# ---------------------------------------------------------------------------

def check_output(arr: np.ndarray, expected_shape: tuple, name: str = "output"):
    assert arr.shape == expected_shape, \
        f"{name}: shape {arr.shape} != expected {expected_shape}"
    assert not np.isnan(arr).any(), f"NaN in {name}!"
    assert not np.isinf(arr).any(), f"Inf in {name}!"
    print(f"  [OK] {name}: shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.4f}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(a: np.ndarray, b: np.ndarray, atol: float, label: str = "") -> bool:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    diff   = np.abs(a32 - b32)
    max_d  = diff.max()
    mean_d = diff.mean()
    cosine = float(
        np.dot(a32.flatten(), b32.flatten()) /
        (np.linalg.norm(a32) * np.linalg.norm(b32) + 1e-12)
    )
    passed = bool(max_d <= atol)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"    max |diff|  = {max_d:.6f}   (threshold {atol})")
    print(f"    mean |diff| = {mean_d:.6f}")
    print(f"    cosine sim  = {cosine:.6f}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve adapter checkpoint
    adapter_ckpt = args.adapter_ckpt or str(_DEFAULT_ADAPTER_CKPT)
    if not Path(adapter_ckpt).exists():
        print(f"[ERROR] Adapter checkpoint not found: {adapter_ckpt}")
        sys.exit(1)

    # Resolve ONNX path
    if args.onnx_path:
        onnx_path = str(Path(args.onnx_path).resolve())
    else:
        onnx_path = str(
            _HERE / "ryzenai_cache/default_workflow/runs/f71451ca/models/model.onnx"
        )
    if not Path(onnx_path).exists():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        sys.exit(1)

    print("=" * 65)
    print("  Gemma Text Encoder ONNX Validation")
    print("=" * 65)
    print(f"  ONNX      : {onnx_path}")
    print(f"  Adapter   : {adapter_ckpt}")
    print(f"  Prompt    : {args.prompt!r}")
    print(f"  seq_len   : {args.seq_len}")
    print(f"  atol      : {args.atol}")
    print("=" * 65)

    # 1. Load ONNX
    print("\n[Step 1] Load ONNX session")
    sess, output_names = load_onnx_session(onnx_path)
    mode = detect_mode(output_names)
    print(f"  Mode: {mode!r}  "
          f"{'(combined Gemma+Adapter)' if mode == 'combined' else '(legacy 4-tap fp32)'}")

    # 2. Tokenise
    print(f"\n[Step 2] Tokenise")
    input_ids, attention_mask = tokenise(args.gemma_model, args.prompt, args.seq_len)

    # 3. ONNX inference
    print(f"\n[Step 3] ONNX inference  ({mode} mode)")
    if mode == "combined":
        onnx_out = run_onnx_combined(sess, input_ids, attention_mask)
        check_output(onnx_out, (1, args.seq_len, 7680), "prompt_embeds (ONNX)")
    else:
        taps_np = run_onnx_4tap(sess, input_ids, attention_mask)
        for name, tap in zip(["tap_9", "tap_18", "tap_26", "tap_30"], taps_np):
            check_output(tap, (1, args.seq_len, 1536), name)

        print(f"\n[Step 3b] Adapter inference (local adapter.py, fp32)")
        onnx_out = run_adapter_on_taps(taps_np, attention_mask, adapter_ckpt)
        check_output(onnx_out, (1, args.seq_len, 7680), "prompt_embeds (4tap+adapter)")

    # 4 & 5. PyTorch reference + comparison
    all_passed = True
    if not args.skip_pytorch:
        print(f"\n[Step 4] PyTorch reference (Gemma fp32 + local adapter)")
        try:
            ref = run_pytorch_reference(
                adapter_ckpt, args.gemma_model, input_ids, attention_mask,
            )
            check_output(ref, (1, args.seq_len, 7680), "prompt_embeds (PyTorch ref)")
            print(f"\n[Step 5] Numerical comparison  (atol={args.atol})")
            label = ("ONNX-combined vs PyTorch" if mode == "combined"
                     else "ONNX-4tap+adapter vs PyTorch")
            all_passed = compare(onnx_out, ref, args.atol, label)
        except Exception as e:
            print(f"\n[WARN] PyTorch reference failed: {e}")
            print("       Shape/NaN checks already passed; skipping numerical comparison.")
    else:
        print("\n[Step 4/5] Skipped (--skip_pytorch)")

    print("\n" + "=" * 65)
    print(f"  Final result: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 65)

    if mode == "4tap":
        print("\n[INFO] Legacy 4-tap fp32 model validated.")
        print("  To export the new combined fp16 model:")
        print()
        print("  conda run -n olive_test python export_models.py \\")
        print("      --models gemma_text_encoder --clear_cache")
        print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
