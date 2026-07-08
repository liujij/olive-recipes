# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Export FLUX.2-klein-4B sub-models to ONNX via Olive.
#
# Usage:
#   python export_models.py [--models transformer vae_decoder text_encoder]
#                           [--model_id <hf_id_or_local_path>]
#                           [--resolutions 1024x1024]
#                           [--output_dir ./output_model]
#
# Output layout:
#   output_model/
#     transformer/dd/replaced.onnx   NPU (RyzenAI)
#     vae_decoder/dd/replaced.onnx   NPU (RyzenAI)
#     text_encoder/model.onnx    fp16 ONNX (ModelBuilder -> VitisGenerateModelSD)
#     tokenizer/                     from pipeline
#     scheduler/                     from pipeline

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from olive.workflows import run as olive_run

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_MODEL_ID   = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_RESOLUTIONS = ["1024x1024"]
ALL_MODELS = ["transformer", "vae_decoder", "text_encoder"]

NON_ONNX_COMPONENTS = ["tokenizer", "tokenizer_2", "scheduler", "feature_extractor"]

# Preprocess-only sub-models are not partitioned: VitisGenerateModelSD still
# emits a dd/replaced.onnx copy alongside the plain model.onnx, but only the
# plain model.onnx is wanted in the assembled output (no dd/ subfolder).
PLAIN_ONNX_MODELS = {"text_encoder"}


def config_dd_env() -> None:
    need_root = not os.environ.get("DD_ROOT") or not os.path.exists(os.environ["DD_ROOT"])
    need_plugins = not os.environ.get("DD_PLUGINS_ROOT") or not os.path.isdir(os.environ["DD_PLUGINS_ROOT"])
    if not need_root and not need_plugins:
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("ryzenai_dynamic_dispatch")
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            if need_root:
                os.environ["DD_ROOT"] = pkg_dir.replace("\\", "/")
                print(f"Set DD_ROOT: {os.environ['DD_ROOT']}")
            if need_plugins:
                bin_dir = os.path.join(pkg_dir, "bin")
                if os.path.isdir(bin_dir):
                    os.environ["DD_PLUGINS_ROOT"] = bin_dir
                    print(f"Set DD_PLUGINS_ROOT: {os.environ['DD_PLUGINS_ROOT']}")
    except Exception as e:
        raise Exception(f"Could not set DD_ROOT or DD_PLUGINS_ROOT: {e}") from e


def _fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def update_config_files(model_id: str | None, resolutions: list[str] | None) -> None:
    for name in ALL_MODELS:
        config_path = SCRIPT_DIR / f"config_{name}.json"
        if not config_path.exists():
            continue
        with config_path.open() as f:
            cfg = json.load(f)

        changed = False
        if model_id is not None and cfg.get("input_model", {}).get("model_path") != model_id:
            cfg["input_model"]["model_path"] = model_id
            changed = True
        if resolutions is not None:
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg and pass_cfg["resolutions"] != resolutions:
                    pass_cfg["resolutions"] = resolutions
                    changed = True

        if changed:
            with config_path.open("w") as f:
                json.dump(cfg, f, indent=4)
            print(f"  [CONFIG] Updated {config_path.name}")


def load_olive_config(submodel_name: str) -> dict:
    config_path = SCRIPT_DIR / f"config_{submodel_name}.json"
    with config_path.open() as f:
        return json.load(f)


def _read_footprint(footprints_dir: Path, submodel_name: str) -> tuple[Path, Path]:
    """Parse footprint.json and return (conversion_path, optimized_path)."""
    from olive.model import ONNXModelHandler

    fp_path = footprints_dir / submodel_name / "footprint.json"
    with fp_path.open() as f:
        footprints = json.load(f)

    # The base graph comes from either OnnxConversion (transformer / vae) or
    # ModelBuilder (text_encoder); any later pass (e.g. VitisGenerateModelSD)
    # is treated as the optimized output.
    conversion_passes = {"onnxconversion", "modelbuilder"}
    conversion_node = None
    optimized_node  = None
    for node in footprints.values():
        from_pass = (node.get("from_pass") or "").lower()
        if from_pass in conversion_passes:
            conversion_node = node
        else:
            optimized_node = node

    if conversion_node is None:
        raise RuntimeError(
            f"Base conversion footprint node not found for '{submodel_name}' in {fp_path}."
        )
    # CPU-only models (text_encoder, vae_encoder) have no optimization pass;
    # the conversion output is the final artifact.
    if optimized_node is None:
        print(f"  [WARN] No optimization pass found for '{submodel_name}'; using conversion output.")
        optimized_node = conversion_node

    def _model_path(node: dict) -> Path:
        cfg = node.get("model_config_data") or node.get("model_config")
        if not cfg:
            raise KeyError(f"Footprint node for '{submodel_name}' missing model_config_data/model_config")
        return Path(ONNXModelHandler(**cfg["config"]).model_path)

    return _model_path(conversion_node), _model_path(optimized_node)


# text_encoder is intentionally omitted: it is a standalone HF model
# (config_text_encoder.json -> input_model.model_path, e.g. Qwen/Qwen3-4B), so
# its config is written from that source (see _save_text_encoder_config), not
# from the diffusers pipeline.
_PIPELINE_COMPONENT_MAP = {
    "transformer": "transformer",
    "vae":         ["vae_encoder", "vae_decoder"],   # VAE covers both encoder & decoder
}


def _save_text_encoder_config(output_dir: Path) -> None:
    """Write config.json / generation_config.json for the text encoder from its
    own HF source (config_text_encoder.json -> input_model.model_path).

    Runs independently of the diffusers pipeline so text-encoder-only exports
    still produce the companion config files.
    """
    dst_dir = output_dir / "text_encoder"
    if not dst_dir.exists():
        return

    cfg = load_olive_config("text_encoder")
    model_path = cfg.get("input_model", {}).get("model_path")
    if not model_path:
        print("  [WARN] text_encoder model_path missing; skipping config.json")
        return

    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_path, trust_remote_code=True).save_pretrained(str(dst_dir))
        print("  [CONFIG]  text_encoder/config.json")
    except Exception as exc:
        print(f"  [WARN] Could not save text_encoder/config.json: {exc}")

    try:
        from transformers import GenerationConfig
        GenerationConfig.from_pretrained(model_path).save_pretrained(str(dst_dir))
        print("  [CONFIG]  text_encoder/generation_config.json")
    except Exception:
        pass


def _save_vae_decoder_bn_stats(pipeline, output_dir: Path) -> None:
    """Extract BN running_mean / running_var from the VAE and save as
    bn.running_x.safetensors next to the vae_decoder ONNX model.

    The RyzenAI runtime loads these stats separately at inference time because
    the ONNX graph does not carry them as initializers.

    Strategy: scan the full VAE state_dict for keys ending in
    'running_mean' / 'running_var', pick the pair with the smallest
    channel dimension (typically 128 for AutoencoderKLFlux2).
    """
    dst = output_dir / "vae_decoder" / "bn.running_x.safetensors"
    if not (output_dir / "vae_decoder").exists() or dst.exists():
        return

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  [WARN] safetensors not installed; skipping bn.running_x.safetensors")
        return

    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return

    sd = vae.state_dict()

    bn_candidates: list[tuple[str, torch.Tensor]] = []
    for key, val in sd.items():
        if key.endswith(".running_mean") and val.ndim == 1:
            prefix = key[: -len(".running_mean")]
            var_key = prefix + ".running_var"
            if var_key in sd:
                bn_candidates.append((prefix, val))

    if not bn_candidates:
        print("  [WARN] No BN running_mean found in VAE state_dict; skipping bn.running_x.safetensors")
        return

    # Use the entry with the smallest channel count.
    prefix, running_mean = min(bn_candidates, key=lambda t: t[1].numel())
    running_var = sd[prefix + ".running_var"]

    tensors = {
        "bn.running_mean": running_mean.detach().to(torch.bfloat16),
        "bn.running_var":  running_var.detach().to(torch.bfloat16),
    }
    save_file(tensors, str(dst))
    print(f"  [SAVE]  vae_decoder/bn.running_x.safetensors  ({prefix}, shape {list(running_mean.shape)})")


def _save_component_configs(pipeline, output_dir: Path) -> None:
    """Save config.json (and generation_config.json) for each ONNX sub-model."""
    import json as _json

    def _write_config(component, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        cfg = getattr(component, "config", None)
        if cfg is None:
            return
        save_fn = getattr(cfg, "save_pretrained", None) or getattr(component, "save_config", None)
        if save_fn:
            try:
                save_fn(str(dst_dir))
                return
            except Exception:
                pass
        to_dict = getattr(cfg, "to_dict", None)
        if to_dict:
            with (dst_dir / "config.json").open("w") as f:
                _json.dump(to_dict(), f, indent=2)

    for attr, targets in _PIPELINE_COMPONENT_MAP.items():
        component = getattr(pipeline, attr, None)
        if component is None:
            continue
        for target in ([targets] if isinstance(targets, str) else targets):
            dst_dir = output_dir / target
            if dst_dir.exists():
                _write_config(component, dst_dir)
                print(f"  [CONFIG]  {target}/config.json")

    _save_vae_decoder_bn_stats(pipeline, output_dir)


def _find_dd_source(optimized_path: Path) -> Path | None:
    """Locate a dd/replaced.onnx directory for a partitioned (NPU) sub-model."""
    candidates = [
        optimized_path / "dd",
        optimized_path / "dynamic" / "dd",
        optimized_path.parent / "dd",
        optimized_path.parent / "dynamic" / "dd",
    ]
    if optimized_path.is_dir():
        if optimized_path.name == "dynamic":
            candidates.append(optimized_path / "dd")
        if optimized_path.name == "dd":
            candidates.append(optimized_path)
    for dd_path in candidates:
        if (dd_path / "replaced.onnx").exists():
            return dd_path
    return None


def _flatten_onnx_to(src_onnx: Path, dst_onnx: Path) -> None:
    """Load a (possibly externally-stored) ONNX and re-save it as a single
    model.onnx + model.onnx.data pair with a self-consistent data reference."""
    import onnx

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src_onnx), load_external_data=True)
    data_name = dst_onnx.name + ".data"  # e.g. model.onnx.data
    stale_data = dst_onnx.parent / data_name
    if stale_data.exists():
        stale_data.unlink()
    onnx.save_model(
        model,
        str(dst_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )


def assemble_output_dir(
    pipeline,
    submodel_names: list[str],
    footprints_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in submodel_names:
        dst_dir = output_dir / name

        try:
            _, optimized_path = _read_footprint(footprints_dir, name)
        except Exception as exc:
            print(f"  [WARN] Could not read footprint for '{name}': {exc}; skipping.")
            continue

        # Preprocess-only models keep a flat model.onnx (no dd/ subfolder).
        # VitisGenerateModelSD writes the self-contained graph under dd/, so we
        # flatten dd/replaced.onnx into <name>/model.onnx (+ model.onnx.data).
        if name in PLAIN_ONNX_MODELS:
            dd_src = _find_dd_source(optimized_path)
            src_onnx = (dd_src / "replaced.onnx") if dd_src is not None else (
                optimized_path if optimized_path.is_file() else optimized_path / "model.onnx"
            )
            if not src_onnx.exists():
                print(f"  [WARN] No ONNX file found for '{name}' at {src_onnx}; skipping.")
                continue
            shutil.rmtree(dst_dir, ignore_errors=True)
            _flatten_onnx_to(src_onnx, dst_dir / "model.onnx")
            print(f"  [COPY CPU]  {name} → {dst_dir / 'model.onnx'}")
            continue

        dd_src = _find_dd_source(optimized_path)

        if dd_src is not None:
            if dd_src.parent.name == "dynamic":
                dst_path = dst_dir / "dynamic" / "dd"
            else:
                dst_path = dst_dir / "dd"
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(dd_src, dst_path)
            print(f"  [COPY NPU] {name} → {dst_path}")
        else:
            # CPU / plain ONNX: copy model.onnx and external data file only.
            onnx_file = optimized_path if optimized_path.is_file() else optimized_path / "model.onnx"
            if not onnx_file.exists():
                print(f"  [WARN] No ONNX file found for '{name}' at {onnx_file}; skipping.")
                continue
            # Start from a clean directory so stale artifacts (e.g. a previous
            # dd/ subfolder) do not linger in the output.
            shutil.rmtree(dst_dir, ignore_errors=True)
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_file, dst_dir / "model.onnx")
            for companion in onnx_file.parent.iterdir():
                if companion == onnx_file or companion.is_dir():
                    continue
                if companion.suffix == ".data" or companion.name.startswith(onnx_file.stem + "."):
                    shutil.copy2(companion, dst_dir / companion.name)
            print(f"  [COPY CPU]  {name} → {dst_dir / 'model.onnx'}")

    # text_encoder config comes from its own HF source, independent of the
    # diffusers pipeline, so it is always written when text_encoder was exported.
    if "text_encoder" in submodel_names:
        _save_text_encoder_config(output_dir)

    # Pipeline-derived metadata (component configs, tokenizers, model_index.json)
    # is only available when the diffusers pipeline was loaded.
    if pipeline is None:
        print(f"\n  Sub-models assembled at: {output_dir}")
        return

    _save_component_configs(pipeline, output_dir)

    for attr in NON_ONNX_COMPONENTS:
        component = getattr(pipeline, attr, None)
        if component is None:
            continue
        save_fn = getattr(component, "save_pretrained", None)
        if save_fn is None:
            continue
        dest = output_dir / attr
        dest.mkdir(parents=True, exist_ok=True)
        save_fn(str(dest))
        print(f"  [SAVE]  {attr} → {dest}")

    # Write the top-level model_index.json so the directory is recognised
    # as a Diffusers pipeline by downstream loaders.
    pipeline.save_config(str(output_dir))
    print("  [SAVE]  model_index.json")

    print(f"\n  Pipeline assembled at: {output_dir}")


def optimize(args) -> dict[str, bool]:
    config_dd_env()
    model_id   = args.model_id
    output_dir = Path(args.output_dir).resolve()

    # All sub-models (including text_encoder, which now uses the
    # VitisGenerateModelSD model_generation pass) are exported through olive_run
    # and assembled from their footprints. The diffusers pipeline only provides
    # component configs / tokenizers / model_index.json, so it is loaded when any
    # pipeline component (transformer / VAE) is requested.
    pipeline_models = [m for m in args.models if m != "text_encoder"]

    pipeline = None
    if pipeline_models:
        print(f"\n[PIPELINE] Loading Flux2KleinPipeline from '{model_id}' ...")
        from diffusers import Flux2KleinPipeline
        pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

        t_cfg   = pipeline.transformer.config
        vae_cfg = pipeline.vae.config
        print(f"  Transformer : in_channels={t_cfg.in_channels}, "
              f"joint_attention_dim={t_cfg.joint_attention_dim}, "
              f"num_layers={t_cfg.num_layers}")
        print(f"  VAE         : latent_channels={vae_cfg.latent_channels}, "
              f"scaling_factor={getattr(vae_cfg, 'scaling_factor', 'N/A')}")

    results: dict[str, bool] = {}
    total_t0 = time.monotonic()

    for submodel_name in args.models:
        print(f"\n{'=' * 60}\n  Exporting: {submodel_name}\n{'=' * 60}")
        t0 = time.monotonic()
        olive_config = load_olive_config(submodel_name)
        try:
            olive_run(olive_config)
            success = True
        except Exception as exc:
            print(f"\n[ERROR] {submodel_name} export failed: {exc}")
            success = False
        elapsed = time.monotonic() - t0
        results[submodel_name] = success
        print(f"\n  [{'OK' if success else 'FAILED'}]  {submodel_name}  ({_fmt_seconds(elapsed)})")

    total_elapsed = time.monotonic() - total_t0

    print(f"\n{'=' * 60}\n  Assembling output directory ...\n{'=' * 60}")
    assemble_output_dir(pipeline, args.models, SCRIPT_DIR / "footprints", output_dir)

    if pipeline is not None:
        del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  Export Summary\n{'=' * 60}")
    for name in args.models:
        print(f"  {'OK    ' if results.get(name) else 'FAILED'}  {name}")
    print(f"{'─' * 60}")
    print(f"  Total time : {_fmt_seconds(total_elapsed)}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    return results


def parse_args(raw_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B sub-models to ONNX via Olive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_models.py\n"
            "  python export_models.py --models transformer\n"
            "  python export_models.py --model_id /local/path/to/model\n"
            "  python export_models.py --output_dir /data/flux2_klein_onnx"
        ),
    )
    parser.add_argument(
        "--model_id", default=None, type=str,
        help=(
            "HuggingFace model ID or local path. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{DEFAULT_MODEL_ID}')."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None, metavar="MODEL",
        help=f"Sub-models to export (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=None, metavar="WxH",
        help=(
            "Target resolutions for VitisGenerateModelSD. "
            "When provided, writes back to all config_*.json. "
            f"Default: value in config_*.json (initially '{' '.join(DEFAULT_RESOLUTIONS)}')."
        ),
    )
    parser.add_argument(
        "--output_dir", default=str(SCRIPT_DIR / "output_model"), type=str,
        help="Assembled pipeline output directory. Default: <script_dir>/output_model",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None) -> None:
    args = parse_args(raw_args)

    if args.models:
        args.models = [m for m in ALL_MODELS if m in args.models]
    else:
        args.models = list(ALL_MODELS)

    if args.model_id is not None or args.resolutions is not None:
        print("\n[CONFIG] Syncing config_*.json ...")
        update_config_files(args.model_id, args.resolutions)

    if args.model_id is None:
        first_cfg_path = SCRIPT_DIR / f"config_{args.models[0]}.json"
        with first_cfg_path.open() as f:
            args.model_id = json.load(f)["input_model"]["model_path"]

    if args.resolutions is None:
        args.resolutions = DEFAULT_RESOLUTIONS
        for name in args.models:
            with (SCRIPT_DIR / f"config_{name}.json").open() as f:
                cfg = json.load(f)
            for pass_cfg in cfg.get("passes", {}).values():
                if "resolutions" in pass_cfg:
                    args.resolutions = pass_cfg["resolutions"]
                    break
            else:
                continue
            break

    print("=" * 60)
    print("  FLUX.2-klein-4B  —  Olive ONNX Export")
    print("=" * 60)
    print(f"  model_id    : {args.model_id}")
    print(f"  sub-models  : {', '.join(args.models)}")
    print(f"  resolutions : {', '.join(args.resolutions)}")
    print(f"  output_dir  : {args.output_dir}")
    print("=" * 60)

    results = optimize(args)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
