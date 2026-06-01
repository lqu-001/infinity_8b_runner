"""Run official GenEval object-detection scoring on Infinity-8B outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from infinity_runner import GENEVAL_REPO, ensure_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score GenEval-format image folders with the official GenEval evaluator.",
    )
    parser.add_argument(
        "imagedir",
        type=Path,
        help="Root folder produced by geneval_generate.py (contains 00000/, 00001/, ...).",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("outputs/geneval/results.jsonl"),
        help="Per-image JSONL results from GenEval.",
    )
    parser.add_argument(
        "--detector-dir",
        type=Path,
        default=Path("weights/geneval_detector"),
        help="Directory containing mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable for GenEval (use your geneval conda env if mmdet is not in the Infinity venv).",
    )
    parser.add_argument("--skip-summary", action="store_true", help="Only write results.jsonl, do not print summary.")
    return parser.parse_args()


def download_detector(detector_dir: Path) -> None:
    detector_dir.mkdir(parents=True, exist_ok=True)
    script = GENEVAL_REPO / "evaluation" / "download_models.sh"
    ensure_exists(script, "GenEval download_models.sh not found. Run setup_geneval.ps1")

    if sys.platform == "win32":
        pth = detector_dir / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
        if pth.is_file():
            return
        url = (
            "https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
        )
        print(f"Downloading Mask2Former weights to {pth} ...")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, pth)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download detector weights. Download manually to:\n  {pth}\nfrom:\n  {url}"
            ) from exc
        return

    subprocess.run(["bash", str(script), str(detector_dir)], check=True)


def main() -> None:
    args = parse_args()
    ensure_exists(GENEVAL_REPO, "GenEval repo not found. Run setup_geneval.ps1")
    ensure_exists(args.imagedir, "Image directory not found")

    detector_pth = args.detector_dir / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    if not detector_pth.is_file():
        download_detector(args.detector_dir)

    python = args.python or sys.executable
    evaluate_script = GENEVAL_REPO / "evaluation" / "evaluate_images.py"
    summary_script = GENEVAL_REPO / "evaluation" / "summary_scores.py"
    ensure_exists(evaluate_script, "GenEval evaluate_images.py not found")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GENEVAL_REPO) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running GenEval detector scoring on {args.imagedir} ...")
    subprocess.run(
        [
            python,
            str(evaluate_script),
            str(args.imagedir),
            "--outfile",
            str(args.outfile),
            "--model-path",
            str(args.detector_dir),
        ],
        cwd=GENEVAL_REPO,
        env=env,
        check=True,
    )
    print(f"Wrote {args.outfile.resolve()}")

    if not args.skip_summary:
        print()
        subprocess.run([python, str(summary_script), str(args.outfile)], cwd=GENEVAL_REPO, check=True)


if __name__ == "__main__":
    main()
