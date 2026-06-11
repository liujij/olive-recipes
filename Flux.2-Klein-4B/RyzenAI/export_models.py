# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Export FLUX.2-klein-4B sub-models to ONNX for Ryzen AI.
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
#     text_encoder/model.onnx        CPU ONNX (prompt_embeds, MatMulNBits INT4)
#     tokenizer/                     from pipeline
#     scheduler/                     from pipeline

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from olive.workflows import run as olive_run

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_RESOLUTIONS = ["1024x1024"]
ALL_MODELS = ["transformer", "vae_decoder", "text_encoder"]

NON_ONNX_COMPONENTS = ["tokenizer", "tokenizer_2", "scheduler", "feature_extractor"]

STAGED_DIR = SCRIPT_DIR / "staged"
STAGING_MARKER = ".staged_from"
TEXT_ENCODER_WEIGHT_GLOBS = ("*.safetensors", "*.json")


def set_dd_env() -> None:
    if os.environ.get("DD_PLUGINS_ROOT"):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("ryzenai_dynamic_dispatch")
        if spec and spec.origin:
            dd_root = os.environ.get("DD_ROOT")
            if not dd_root or not os.path.exists(dd_root):
                os.environ["DD_ROOT"] = os.path.dirname(spec.origin).replace("\\", "/")
            bin_dir = os.path.join(os.path.dirname(spec.origin), "bin")
            if os.path.isdir(bin_dir):
                os.environ["DD_PLUGINS_ROOT"] = bin_dir
    except Exception:
        pass


def _fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink large weight files when possible; fall back to copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _staging_dir_for_pipeline(pipeline_root: Path) -> Path:
    digest = hashlib.sha256(str(pipeline_root).encode()).hexdigest()[:12]
    return STAGED_DIR / f"text_encoder_{digest}"


def resolve_pipeline_root(model_id: str | Path) -> Path:
    """Return the diffusers pipeline root for Flux2KleinPipeline loading."""
    path = Path(model_id).resolve()
    if (path / "model_index.json").exists():
        return path
    if (path / "text_encoder" / "config.json").exists():
        return path

    marker = path / STAGING_MARKER
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())

    raise ValueError(
        f"Cannot resolve diffusers pipeline root from '{path}'. "
        "Pass --model_id pointing to the FLUX.2-klein-4B pipeline directory."
    )


def stage_text_encoder_bundle(pipeline_root: str | Path) -> Path:
    """Assemble text_encoder weights + tokenizer into one HF-style directory.

    Diffusers pipelines keep ``text_encoder/`` and ``tokenizer/`` as siblings.
    ModelBuilder expects a single checkpoint directory with ``config.json``,
    weight shards, and tokenizer files together.
    """
    pipeline_root = Path(pipeline_root).resolve()
    text_encoder_src = pipeline_root / "text_encoder"
    tokenizer_src = pipeline_root / "tokenizer"

    if not text_encoder_src.is_dir():
        raise FileNotFoundError(f"Missing text_encoder directory: {text_encoder_src}")
    if not (text_encoder_src / "config.json").exists():
        raise FileNotFoundError(f"Missing text_encoder config: {text_encoder_src / 'config.json'}")
    if not tokenizer_src.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_src}")

    dest = _staging_dir_for_pipeline(pipeline_root)
    marker = dest / STAGING_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == str(pipeline_root):
        if (dest / "config.json").exists() and (dest / "tokenizer.json").exists():
            print(f"  [STAGE] Reusing staged text_encoder bundle: {dest}")
            return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for pattern in TEXT_ENCODER_WEIGHT_GLOBS:
        for src_file in sorted(text_encoder_src.glob(pattern)):
            if src_file.is_file():
                _link_or_copy(src_file, dest / src_file.name)

    for src_file in sorted(tokenizer_src.iterdir()):
        if src_file.is_file():
            _link_or_copy(src_file, dest / src_file.name)

    marker.write_text(str(pipeline_root), encoding="utf-8")
    print(f"  [STAGE] Assembled text_encoder bundle: {dest}")
    return dest


def prepare_text_encoder_for_export(pipeline_root: str | Path) -> Path:
    """Stage a flat HF checkpoint directory for text encoder export."""
    return stage_text_encoder_bundle(pipeline_root)


def _write_text_encoder_footprint(footprint_dir: Path) -> None:
    """Write a minimal footprint so assemble_output_dir can copy model.onnx."""
    footprint_dir.mkdir(parents=True, exist_ok=True)
    node_id = "matmulnbits_export"
    footprint = {
        node_id: {
            "parent_model_id": None,
            "model_id": node_id,
            "model_config_data": {
                "type": "onnxmodel",
                "config": {
                    "model_path": str(footprint_dir),
                    "onnx_file_name": "model.onnx",
                },
            },
            "from_pass": "onnxconversion",
        }
    }
    with (footprint_dir / "footprint.json").open("w", encoding="utf-8") as f:
        json.dump(footprint, f, indent=4)


def _find_latest_model_onnx(search_root: Path) -> Path:
    """Pick the most recently modified ``model.onnx`` under an output tree."""
    candidates = list(search_root.rglob("model.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No model.onnx found under {search_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_fp16_prompt_embed_modelbuilder(staged_model_dir: Path, run_config: Path) -> Path:
    """Run ModelBuilder (fp16, prompt_embeds) and return path to ``model.onnx``."""
    run_config = run_config.resolve()
    if not run_config.is_file():
        raise FileNotFoundError(f"ModelBuilder run config not found: {run_config}")

    with run_config.open(encoding="utf-8") as f:
        cfg = json.load(f)

    input_model = cfg.setdefault("input_model", {})
    input_model["model_path"] = str(staged_model_dir.resolve())
    load_kw = input_model.setdefault("load_kwargs", {})
    load_kw.setdefault("trust_remote_code", True)

    out_rel = cfg.get("output_dir", "footprints/text_encoder_fp16_mb")
    out_dir = Path(out_rel)
    if not out_dir.is_absolute():
        out_dir = (SCRIPT_DIR / out_dir).resolve()
    cfg["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = out_dir / "_export_models_fp16_run_config.json"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [FP16] ModelBuilder (prompt_embeds): effective config → {sidecar}")

    olive_run(cfg)
    fp16_onnx = _find_latest_model_onnx(out_dir)
    print(f"  [FP16] ONNX: {fp16_onnx}")
    return fp16_onnx


def export_text_encoder_matmulnbits(staged_model_dir: Path) -> Path:
    """Text encoder: ModelBuilder fp16 (recipe JSON) → MatMulNBits INT4."""
    from text_encoder_matmulnbits import export_prompt_embeds_matmulnbits

    recipe = (SCRIPT_DIR / "recipes" / "qwen3-4b-fp16-prompt-embeds-modelbuilder.json").resolve()
    if not recipe.is_file():
        raise FileNotFoundError(
            f"Text encoder fp16 recipe not found: {recipe}. "
            "Restore recipes/ in this package."
        )
    print(f"  [TEXT_ENCODER] Using fp16 recipe: {recipe}")
    resolved_fp16 = _run_fp16_prompt_embed_modelbuilder(staged_model_dir, recipe)

    footprint_dir = SCRIPT_DIR / "footprints" / "text_encoder"
    output_onnx = footprint_dir / "model.onnx"
    export_prompt_embeds_matmulnbits(staged_model_dir, output_onnx, fp16_onnx_path=resolved_fp16)
    _write_text_encoder_footprint(footprint_dir)
    return output_onnx


def _config_paths_for_update(models: list[str]) -> list[Path]:
    paths: list[Path] = []
    for name in models:
        if name == "text_encoder":
            continue
        paths.append(SCRIPT_DIR / f"config_{name}.json")
    return paths


def update_config_files(
    model_id: str | None,
    resolutions: list[str] | None,
    models: list[str],
) -> None:
    for config_path in _config_paths_for_update(models):
        if not config_path.exists():
            continue
        with config_path.open() as f:
            cfg = json.load(f)

        changed = False
        if model_id is not None:
            if cfg.get("input_model", {}).get("model_path") != model_id:
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
    """Load Olive workflow JSON for NPU sub-models (transformer, vae_decoder)."""
    if submodel_name == "text_encoder":
        raise ValueError(
            "text_encoder is exported via ModelBuilder fp16 → MatMulNBits, not a static config JSON."
        )
    config_path = SCRIPT_DIR / f"config_{submodel_name}.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_footprint(footprints_dir: Path, submodel_name: str) -> tuple[Path, Path]:
    """Parse footprint.json and return (conversion_path, optimized_path)."""
    from olive.model import ONNXModelHandler

    fp_path = footprints_dir / submodel_name / "footprint.json"
    with fp_path.open() as f:
        footprints = json.load(f)

    conversion_node = None
    optimized_node = None
    modelbuilder_node = None
    for node in footprints.values():
        from_pass = (node.get("from_pass") or "").lower()
        if from_pass == "onnxconversion":
            conversion_node = node
        elif from_pass == "modelbuilder":
            modelbuilder_node = node
        else:
            optimized_node = node

    if conversion_node is None:
        if modelbuilder_node is not None:
            conversion_node = modelbuilder_node
            optimized_node = modelbuilder_node
        elif optimized_node is not None:
            print(
                f"  [WARN] OnnxConversion footprint node not found for '{submodel_name}'; "
                "using last optimization pass output."
            )
            conversion_node = optimized_node
        else:
            raise RuntimeError(
                f"OnnxConversion footprint node not found for '{submodel_name}' in {fp_path}."
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


_PIPELINE_COMPONENT_MAP = {
    "transformer": "transformer",
    "vae": ["vae_encoder", "vae_decoder"],  # VAE covers both encoder & decoder
    "text_encoder": "text_encoder",
}


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
        "bn.running_var": running_var.detach().to(torch.bfloat16),
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
                if attr == "text_encoder":
                    gen_cfg = getattr(component, "generation_config", None)
                    save_gen = getattr(gen_cfg, "save_pretrained", None) if gen_cfg else None
                    if save_gen:
                        try:
                            save_gen(str(dst_dir))
                        except Exception:
                            pass
                print(f"  [CONFIG]  {target}/config.json")

    _save_vae_decoder_bn_stats(pipeline, output_dir)


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
                dd_src = dd_path
                break
        else:
            dd_src = None

        if dd_src is not None:
            dst_path = dst_dir / ("dd" if dd_src.name == "dd" else "dynamic" / "dd")
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(dd_src, dst_path)
            print(f"  [COPY NPU] {name} → {dst_path}")
        else:
            # CPU / plain ONNX: copy model.onnx and external data file only.
            onnx_file = optimized_path if optimized_path.is_file() else optimized_path / "model.onnx"
            if not onnx_file.exists():
                print(f"  [WARN] No ONNX file found for '{name}' at {onnx_file}; skipping.")
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_file, dst_dir / "model.onnx")
            for companion in onnx_file.parent.iterdir():
                if companion == onnx_file or companion.is_dir():
                    continue
                if companion.suffix == ".data" or companion.name.startswith(onnx_file.stem + "."):
                    shutil.copy2(companion, dst_dir / companion.name)
            print(f"  [COPY CPU]  {name} → {dst_dir / 'model.onnx'}")

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
    model_id = args.model_id
    output_dir = Path(args.output_dir).resolve()

    print(f"\n[PIPELINE] Loading Flux2KleinPipeline from '{model_id}' ...")
    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    t_cfg = pipeline.transformer.config
    vae_cfg = pipeline.vae.config
    print(
        f"  Transformer : in_channels={t_cfg.in_channels}, "
        f"joint_attention_dim={t_cfg.joint_attention_dim}, "
        f"num_layers={t_cfg.num_layers}"
    )
    print(
        f"  VAE         : latent_channels={vae_cfg.latent_channels}, "
        f"scaling_factor={getattr(vae_cfg, 'scaling_factor', 'N/A')}"
    )

    results: dict[str, bool] = {}
    total_t0 = time.monotonic()

    for submodel_name in args.models:
        print(f"\n{'=' * 60}\n  Exporting: {submodel_name}\n{'=' * 60}")
        t0 = time.monotonic()
        try:
            if submodel_name == "text_encoder":
                print("  text_encoder: ModelBuilder fp16 → MatMulNBits INT4 (genai)")
                staged_path = prepare_text_encoder_for_export(resolve_pipeline_root(model_id))
                export_text_encoder_matmulnbits(staged_path)
                success = True
            else:
                olive_config = load_olive_config(submodel_name)
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
        description="Export FLUX.2-klein-4B sub-models to ONNX for Ryzen AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export_models.py\n"
            "  python export_models.py --models transformer\n"
            "  python export_models.py --model_id /local/path/to/model\n"
            "  python export_models.py --models text_encoder\n"
            "  python export_models.py --output_dir /data/flux2_klein_onnx"
        ),
    )
    parser.add_argument(
        "--model_id",
        default=None,
        type=str,
        help=(
            "HuggingFace model ID or local path. "
            "When provided, writes back to config_transformer.json and config_vae_decoder.json. "
            f"Default: value in those configs (initially '{DEFAULT_MODEL_ID}'), "
            "or '{DEFAULT_MODEL_ID}' when exporting text_encoder alone without --model_id."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=None,
        metavar="MODEL",
        help=f"Sub-models to export (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=None,
        metavar="WxH",
        help=(
            "Target resolutions for VitisGenerateModelSD. "
            "When provided, writes back to config_transformer.json and config_vae_decoder.json. "
            f"Default: value in those configs (initially '{' '.join(DEFAULT_RESOLUTIONS)}')."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "output_model"),
        type=str,
        help="Assembled pipeline output directory. Default: <script_dir>/output_model",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None) -> None:
    set_dd_env()
    args = parse_args(raw_args)

    if args.models:
        args.models = [m for m in ALL_MODELS if m in args.models]
    else:
        args.models = list(ALL_MODELS)

    if args.model_id is not None or args.resolutions is not None:
        print("\n[CONFIG] Syncing config_*.json ...")
        update_config_files(args.model_id, args.resolutions, args.models)

    if args.model_id is None:
        if set(args.models) <= {"text_encoder"}:
            args.model_id = str(DEFAULT_MODEL_ID)
        else:
            ref_model = next((m for m in args.models if m != "text_encoder"), args.models[0])
            cfg_path = SCRIPT_DIR / f"config_{ref_model}.json"
            with cfg_path.open(encoding="utf-8") as f:
                args.model_id = json.load(f)["input_model"]["model_path"]

    pipeline_root = resolve_pipeline_root(args.model_id)

    if "text_encoder" in args.models:
        print("\n[STAGE] Preparing flat text_encoder bundle ...")
        staged_path = prepare_text_encoder_for_export(pipeline_root)
        print(f"  text_encoder bundle: {staged_path}")
        print(f"  pipeline model_id  : {pipeline_root}")
        args.model_id = str(pipeline_root)

    if args.resolutions is None:
        args.resolutions = DEFAULT_RESOLUTIONS
        for name in args.models:
            if name == "text_encoder":
                continue
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
    print("  FLUX.2-klein-4B  —  Ryzen AI ONNX Export")
    print("=" * 60)
    print(f"  model_id    : {args.model_id}")
    print(f"  sub-models  : {', '.join(args.models)}")
    if "text_encoder" in args.models:
        print("  text_encoder: ModelBuilder fp16 → MatMulNBits INT4")
        print("  text_encoder fp16: recipes/qwen3-4b-fp16-prompt-embeds-modelbuilder.json")
    print(f"  resolutions : {', '.join(args.resolutions)}")
    print(f"  output_dir  : {args.output_dir}")
    print("=" * 60)

    results = optimize(args)
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
