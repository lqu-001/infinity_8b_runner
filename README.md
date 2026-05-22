# Infinity-8B Official Inference Runner

This project wraps the official FoundationVision Infinity code for Infinity-8B image generation.

Official sources:

- Code: https://github.com/FoundationVision/Infinity
- Weights: https://huggingface.co/FoundationVision/Infinity

## Hardware Notes

Infinity-8B is very large:

- `infinity_8b_weights`: about 33.5 GB
- `infinity_vae_d56_f8_14_patchify.pth`: about 1.22 GB
- `google/flan-t5-xl`: several GB

For 1024x1024 inference, expect an 80 GB class GPU to be the realistic path. The official code is Linux/CUDA oriented. Windows native installs may struggle with `flash_attn`; WSL2 or Linux is recommended.

If you only want to verify the pipeline, use `--resolution 512`, but the transformer weights are still large.

## Setup

From the repo root:

```powershell
cd infinity_8b_runner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install download helper:

```powershell
pip install -r requirements-download.txt
```

Clone the official Infinity repo:

```powershell
.\setup_official_infinity.ps1
```

Install official dependencies:

```powershell
cd third_party\Infinity
pip install -r requirements.txt
cd ..\..
```

For Linux/WSL, the same setup is:

```bash
cd infinity_8b_runner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-download.txt
mkdir -p third_party
git clone https://github.com/FoundationVision/Infinity.git third_party/Infinity
pip install -r third_party/Infinity/requirements.txt
```

## Download Weights

Download the official 1024x1024 Infinity-8B weights, VAE, and Flan-T5-XL:

```powershell
python download_weights.py --resolution 1024
```

For the 512x512 Infinity-8B checkpoint:

```powershell
python download_weights.py --resolution 512
```

If `weights\flan-t5-xl-official` already exists, skip the T5 download:

```powershell
python download_weights.py --resolution 1024 --skip-t5
```

Expected layout:

```text
infinity_8b_runner/
  third_party/Infinity/
  weights/
    infinity_8b_weights/
    infinity_vae_d56_f8_14_patchify.pth
    flan-t5-xl-official/
```

## Generate

Run 1024x1024 generation:

```powershell
python generate_infinity_8b.py --resolution 1024 --prompt "alien spaceship enterprise" --output outputs\alien.png
```

Run 512x512 generation:

```powershell
python generate_infinity_8b.py --resolution 512 --prompt "a corgi wearing sunglasses" --output outputs\corgi.png
```

Useful options:

```powershell
python generate_infinity_8b.py --prompt "a robot holding a huge eggplant, sunny nature background" --cfg 3 --tau 0.5 --seed 123 --positive-prompt
```

The wrapper calls the official `tools/run_infinity.py` with the same 8B settings used in `tools/interactive_infer_8b.ipynb`:

```text
model_type = infinity_8b
checkpoint_type = torch_shard
vae_type = 14
apply_spatial_patchify = 1
bf16 = 1
```

## Common Issues

- `ModuleNotFoundError: No module named 'infinity'`: run through `generate_infinity_8b.py`; it sets `PYTHONPATH` for the official repo.
- Missing `flan-t5-xl`: run `python download_weights.py`, or place the model under `weights/flan-t5-xl-official`.
- CUDA OOM: Infinity-8B is likely too large for 24 GB cards. Try 512 first, reduce other GPU usage, or use an 80 GB class GPU.
- `flash_attn` install errors on Windows: use Linux/WSL2 or a CUDA Linux environment.
