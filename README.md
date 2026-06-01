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
cd D:\proj\infinity_8b
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
cd infinity_8b
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
infinity_8b/
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

## GenEval benchmark

[GenEval](https://github.com/djghosh13/geneval) is a common T2I compositional benchmark (~553 prompts, 4 images each). This repo supports **生成 → 评测 → 打分**（打分由官方 `summary_scores.py` 在评测结束后自动打印）。

### 完整流程（按顺序做一遍即可）

在仓库根目录 `infinity_8b/` 下（建议先 `python -m venv .venv` 并激活）：

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| 1 | 见上文 **Setup** | 克隆 `third_party/Infinity`，安装其 `requirements.txt` |
| 2 | `python download_weights.py --resolution 1024` | 下载 8B / VAE / T5 权重到 `weights/` |
| 3 | `.\setup_geneval.ps1` | 克隆 `third_party/geneval` 与 prompt 列表 |
| 4a | `.\geneval_run_1024.ps1` | **一键**：1024 生成 + 评测 + 终端汇总分数 |
| 4b | （可选）分步 | 生成见下节；评测见 **Evaluate and score** |

生成产物：`outputs/geneval/00000/samples/*.png`  
评测产物：`outputs/geneval/results.jsonl`  
终端会打印各 task 准确率及 **Overall score**（GenEval 总分）。

**环境注意：**

- **生成**需要 Infinity 环境 + 大显存 GPU（1024 建议约 80GB 级；见 Hardware Notes）。
- **评测**依赖 `mmdet` + CUDA，官方 GenEval 面向 **Linux/WSL**。在 Windows 上通常只能完成步骤 1–3 与生成；评测请在 WSL2 中安装 [geneval/environment.yml](https://github.com/djghosh13/geneval/blob/main/environment.yml)，然后：
  `python geneval_run_1024.py --skip-generate --geneval-python /path/to/geneval/bin/python`

`geneval_run_1024.ps1` 会在缺少 prompt 时自动执行 `setup_geneval.ps1`，但**不会**代替你完成 Infinity 安装与权重下载。

### Setup GenEval prompts

```powershell
.\setup_geneval.ps1
```

This clones `third_party/geneval` (includes `prompts/evaluation_metadata.jsonl`).

### One-shot: generate + score @ 1024

```powershell
.\geneval_run_1024.ps1
```

Or:

```powershell
python geneval_run_1024.py
```

This runs the full pipeline at **1024×1024**: all GenEval prompts → `outputs/geneval/` → `outputs/geneval/results.jsonl` + printed summary scores.

If evaluation uses a separate GenEval conda env (Linux/WSL):

```powershell
python geneval_run_1024.py --geneval-python /path/to/geneval-env/bin/python
```

Useful flags: `--resume` (continue generation), `--skip-generate` / `--skip-eval`, `--start` / `--end` for a subset.

### Generate images only (Infinity-8B)

Loads the model once and writes the layout expected by GenEval:

```text
outputs/geneval/
  00000/
    metadata.jsonl
    samples/
      0000.png ... 0003.png
  00001/
    ...
```

```powershell
python geneval_generate.py --outdir outputs\geneval --resolution 1024 --n-samples 4 --seed 42
```

Useful flags:

```powershell
# Subset or resume interrupted runs
python geneval_generate.py --start 0 --end 10
python geneval_generate.py --resume

# Match single-image defaults
python geneval_generate.py --cfg 3 --tau 0.5 --positive-prompt
```

For a full run at 1024×1024, plan for long wall time and large disk usage (~553 × 4 images).

### Evaluate and score

GenEval scoring uses Mask2Former + MMDetection and **requires CUDA Linux** (same as upstream GenEval). Use a separate conda env from [geneval/environment.yml](https://github.com/djghosh13/geneval/blob/main/environment.yml) if your Infinity venv lacks `mmdet`.

```powershell
# Download detector weights (or use geneval/evaluation/download_models.sh on Linux)
python geneval_evaluate.py outputs\geneval --outfile outputs\geneval\results.jsonl --detector-dir weights\geneval_detector
```

If GenEval deps live in another Python:

```powershell
python geneval_evaluate.py outputs\geneval --python "C:\path\to\geneval-env\python.exe"
```

The script prints per-task accuracy and the overall GenEval score (average over task groups), matching `evaluation/summary_scores.py`.

## Common Issues

- `ModuleNotFoundError: No module named 'infinity'`: run through `generate_infinity_8b.py` or `geneval_generate.py`; they add the official repo to `PYTHONPATH`.
- Missing `flan-t5-xl`: run `python download_weights.py`, or place the model under `weights/flan-t5-xl-official`.
- CUDA OOM: Infinity-8B is likely too large for 24 GB cards. Try 512 first, reduce other GPU usage, or use an 80 GB class GPU.
- `flash_attn` install errors on Windows: use Linux/WSL2 or a CUDA Linux environment.
- GenEval `evaluate_images.py` fails on Windows: run evaluation under WSL2/Linux with the official GenEval environment; generation on Windows is fine.
