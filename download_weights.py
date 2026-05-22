import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official Infinity-8B inference weights.")
    parser.add_argument(
        "--resolution",
        choices=("512", "1024"),
        default="1024",
        help="Download the 512x512 or 1024x1024 Infinity-8B transformer shards.",
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help="Do not download google/flan-t5-xl. Use this if it already exists under weights/flan-t5-xl-official.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    transformer_dir = "infinity_8b_512x512_weights" if args.resolution == "512" else "infinity_8b_weights"
    print(f"Downloading Infinity-8B transformer shards: {transformer_dir}")
    snapshot_download(
        repo_id="FoundationVision/Infinity",
        allow_patterns=f"{transformer_dir}/*",
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    print("Downloading Infinity-8B VAE: infinity_vae_d56_f8_14_patchify.pth")
    hf_hub_download(
        repo_id="FoundationVision/Infinity",
        filename="infinity_vae_d56_f8_14_patchify.pth",
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    if not args.skip_t5:
        print("Downloading text encoder: google/flan-t5-xl")
        snapshot_download(
            repo_id="google/flan-t5-xl",
            local_dir=WEIGHTS_DIR / "flan-t5-xl-official",
            local_dir_use_symlinks=False,
        )

    print("done")


if __name__ == "__main__":
    main()
