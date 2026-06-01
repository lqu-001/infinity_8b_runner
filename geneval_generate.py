"""Generate GenEval-format images with Infinity-8B (model loaded once)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infinity_runner import (
    DEFAULT_GENEVAL_METADATA,
    GENEVAL_REPO,
    build_infinity_args,
    ensure_exists,
    generate_image,
    load_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Infinity-8B on GenEval prompts and write GenEval-compatible image folders.",
    )
    parser.add_argument(
        "metadata_file",
        type=Path,
        nargs="?",
        default=DEFAULT_GENEVAL_METADATA,
        help="GenEval evaluation_metadata.jsonl (default: third_party/geneval/prompts/...).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/geneval"),
        help="Root folder for GenEval layout (00000/samples/0000.png, ...).",
    )
    parser.add_argument("--resolution", choices=("512", "1024"), default="1024")
    parser.add_argument("--n-samples", type=int, default=4, help="Images per prompt (GenEval default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; sample i uses seed + i.")
    parser.add_argument("--cfg", default="3", help="Classifier-free guidance for Infinity-8B.")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--positive-prompt", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="First prompt index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last prompt index (exclusive).")
    parser.add_argument("--resume", action="store_true", help="Skip prompts whose samples already exist.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict]:
    ensure_exists(path, "GenEval metadata file not found. Run setup_geneval.ps1")
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def samples_complete(sample_dir: Path, n_samples: int) -> bool:
    if not sample_dir.is_dir():
        return False
    for i in range(n_samples):
        if not (sample_dir / f"{i:04d}.png").is_file():
            return False
    return True


def main() -> None:
    args = parse_args()
    if not GENEVAL_REPO.exists():
        raise FileNotFoundError(
            f"GenEval repo not found at {GENEVAL_REPO}. Run setup_geneval.ps1 first."
        )

    metadatas = load_metadata(args.metadata_file)
    end = len(metadatas) if args.end is None else min(args.end, len(metadatas))
    if args.start < 0 or args.start >= end:
        raise ValueError(f"Invalid range: start={args.start}, end={end}, total={len(metadatas)}")

    infinity_args = build_infinity_args(
        args.resolution,
        cfg=args.cfg,
        tau=args.tau,
        seed=args.seed,
        positive_prompt=args.positive_prompt,
    )
    print(f"Loading Infinity-8B ({args.resolution})...")
    pipeline = load_pipeline(infinity_args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, end):
        metadata = metadatas[index]
        prompt = metadata["prompt"]
        outpath = args.outdir / f"{index:05d}"
        sample_dir = outpath / "samples"

        if args.resume and samples_complete(sample_dir, args.n_samples):
            print(f"[{index:05d}/{end - 1}] skip (complete): {prompt[:80]!r}")
            continue

        outpath.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (outpath / "metadata.jsonl").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        print(f"[{index:05d}/{end - 1}] {prompt!r}")
        for sample_idx in range(args.n_samples):
            image_path = sample_dir / f"{sample_idx:04d}.png"
            if args.resume and image_path.is_file():
                continue
            generate_image(
                pipeline,
                prompt,
                image_path,
                seed=args.seed + sample_idx,
            )

    print(f"Done. Images under {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
