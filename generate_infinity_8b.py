import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO = ROOT / "third_party" / "Infinity"
WEIGHTS_DIR = ROOT / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Infinity-8B text-to-image inference.")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "infinity_8b.png")
    parser.add_argument("--resolution", choices=("512", "1024"), default="1024")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", default="3", help="Classifier-free guidance, e.g. 3 or comma-separated schedule.")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--positive-prompt", action="store_true", help="Enable official positive prompt augmentation.")
    return parser.parse_args()


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def main() -> None:
    args = parse_args()
    ensure_exists(OFFICIAL_REPO / "tools" / "run_infinity.py", "Official Infinity repo not found. Run setup_official_infinity.ps1 first")

    transformer_dir = WEIGHTS_DIR / ("infinity_8b_512x512_weights" if args.resolution == "512" else "infinity_8b_weights")
    pn = "0.25M" if args.resolution == "512" else "1M"
    vae_path = WEIGHTS_DIR / "infinity_vae_d56_f8_14_patchify.pth"
    t5_path = WEIGHTS_DIR / "flan-t5-xl-official"

    ensure_exists(transformer_dir, "Missing Infinity-8B transformer weights. Run download_weights.py")
    ensure_exists(vae_path, "Missing Infinity VAE. Run download_weights.py")
    ensure_exists(t5_path, "Missing flan-t5-xl text encoder. Run download_weights.py")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(OFFICIAL_REPO / "tools" / "run_infinity.py"),
        "--prompt",
        args.prompt,
        "--save_file",
        str(args.output),
        "--cfg",
        args.cfg,
        "--tau",
        str(args.tau),
        "--pn",
        pn,
        "--model_path",
        str(transformer_dir),
        "--cfg_insertion_layer",
        "0",
        "--vae_type",
        "14",
        "--vae_path",
        str(vae_path),
        "--add_lvl_embeding_only_first_block",
        "1",
        "--use_bit_label",
        "1",
        "--model_type",
        "infinity_8b",
        "--rope2d_each_sa_layer",
        "1",
        "--rope2d_normalized_by_hw",
        "2",
        "--use_scale_schedule_embedding",
        "0",
        "--sampling_per_bits",
        "1",
        "--text_encoder_ckpt",
        str(t5_path),
        "--text_channels",
        "2048",
        "--apply_spatial_patchify",
        "1",
        "--h_div_w_template",
        "1.000",
        "--use_flex_attn",
        "0",
        "--enable_positive_prompt",
        "1" if args.positive_prompt else "0",
        "--cache_dir",
        str(ROOT / ".cache"),
        "--enable_model_cache",
        "0",
        "--checkpoint_type",
        "torch_shard",
        "--seed",
        str(args.seed),
        "--bf16",
        "1",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OFFICIAL_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=OFFICIAL_REPO, env=env, check=True)


if __name__ == "__main__":
    main()
