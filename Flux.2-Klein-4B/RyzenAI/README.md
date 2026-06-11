# FLUX.2-klein-4B — Model Optimization for AMD NPU

This folder contains Olive workflow JSON for the **transformer** and **VAE decoder** NPU passes, plus an export script that assembles the full diffusers-style pipeline directory.

The export script (`export_models.py`) handles the full pipeline in
one command: download weights → ONNX conversion → NPU compilation →
assemble a self-contained output directory (ONNX models + tokenizer + scheduler).

The **text encoder** is exported on a fixed **genai** path: Olive ModelBuilder (fp16, `prompt_embeds`) followed by MatMulNBits INT4 quantization in Python (`text_encoder_matmulnbits.py`). It does not use `config_text_encoder*.json`.

## Prerequisites

| Requirement | Notes |
|---|---|
| AMD NPU hardware | Ryzen AI device (NPU required for transformer / VAE decoder) |
| Windows 10/11 (x64) | Tested environment |
| Conda | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda |
| ~40 GB free disk | Weights + ONNX artifacts + Olive cache |
| HuggingFace account | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a gated model — license acceptance required |

## Step 1 — Create the conda environment

```bash
conda create -n flux2_klein_export python=3.12 -y
conda activate flux2_klein_export
```

## Step 2 — Install model dependencies

```bash
git clone https://github.com/microsoft/olive-recipes.git
cd olive-recipes/Flux.2-Klein-4B/RyzenAI
pip install -r requirements_ryzenai_sd.txt
```

## Step 3 — Run the export

```bash
python export_models.py
```

The script will:

1. Download and cache the full pipeline weights from HuggingFace (first run only).
2. Export each sub-model: transformer and VAE decoder via Olive; text encoder via ModelBuilder fp16 → MatMulNBits INT4.
3. Compile the transformer and VAE decoder for AMD NPU using `VitisGenerateModelSD`.
4. Assemble the final pipeline directory, including non-ONNX components
   (tokenizer, scheduler).

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | value stored in `config_transformer.json` / `config_vae_decoder.json`, or HF default when exporting text encoder alone | HuggingFace model ID or local path. Written back to those config files when set. |
| `--models` | all | Sub-models to export. Choices: `transformer vae_decoder text_encoder` |
| `--resolutions` | `1024x1024` | NPU compilation resolution(s). Written back to `config_transformer.json` and `config_vae_decoder.json`. |
| `--output_dir` | `./output_model` | Destination for the assembled pipeline directory. |

```bash
# Export only the transformer
python export_models.py --models transformer

# Use a local model directory
python export_models.py --model_id D:/models/FLUX.2-klein-4B

# Change output directory
python export_models.py --output_dir D:/output/flux2_klein

# Export text encoder only
python export_models.py --model_id /path/to/FLUX.2-klein-4B --models text_encoder
```

### Text encoder (genai)

The text encoder export uses `recipes/qwen3-4b-fp16-prompt-embeds-modelbuilder.json` (CUDA EP) for fp16 `prompt_embeds`, then `quantize_matmul_4bits` → `MatMulNBits` (block 128).

Exports `prompt_embeds` with shape `[batch, sequence, 7680]`.

For a **manual** `olive run` with the same JSON (paths not auto-filled):

```bash
cd RyzenAI
# Set input_model.model_path in the JSON to your HF checkpoint directory, or rely on export_models to patch it when run from the script.
olive run --run-config recipes/qwen3-4b-fp16-prompt-embeds-modelbuilder.json
```

MatMul→MatMulNBits conversion only applies to `MatMul` nodes with a constant 2-D weight; graphs dominated by `Gemm` may report zero converted MatMul nodes.

## Output layout

After a successful export, `output_model/` (or your `--output_dir`) will contain:

```
output_model/
├── transformer/
│   ├── dd/
│   │   └── replaced.onnx      ← NPU-compiled
│   └── cache/
├── vae_decoder/
│   ├── dd/
│   │   └── replaced.onnx      ← NPU-compiled
│   └── cache/
├── text_encoder/
│   └── model.onnx             ← CPU ONNX
├── tokenizer/
└── scheduler/
```

> Olive writes intermediate outputs under `footprints/` and caches converted
> models in `cache/`. These can be safely deleted after the final pipeline is
> assembled.
