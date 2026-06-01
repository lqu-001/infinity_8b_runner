"""One-shot GenEval @ 1024: generate Infinity-8B images, then run official scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOLUTION = "1024"
DEFAULT_OUTDIR = ROOT / "outputs" / "geneval"
DEFAULT_RESULTS = DEFAULT_OUTDIR / "results.jsonl"
DEFAULT_DETECTOR = ROOT / "weights" / "geneval_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GenEval full pipeline at 1024×1024: generate then evaluate/score.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="GenEval image root (default: outputs/geneval).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Per-image JSONL output (default: <outdir>/results.jsonl).",
    )
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument(
        "--geneval-python",
        type=str,
        default=None,
        help="Python for evaluation (geneval conda env with mmdet). Defaults to current interpreter.",
    )
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", default="3")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--positive-prompt", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-generate", action="store_true", help="Only run evaluation on existing --outdir.")
    parser.add_argument("--skip-eval", action="store_true", help="Only run generation.")
    return parser.parse_args()


def run_module(module_name: str, argv: list[str]) -> None:
    sys.argv = [module_name, *argv]
    if module_name == "geneval_generate":
        from geneval_generate import main

        main()
    elif module_name == "geneval_evaluate":
        from geneval_evaluate import main

        main()
    else:
        raise ValueError(module_name)


def main() -> None:
    args = parse_args()
    results = args.results or (args.outdir / "results.jsonl")

    print("=" * 60)
    print("GenEval @ 1024 — Infinity-8B")
    print(f"  images:   {args.outdir.resolve()}")
    print(f"  results:  {results.resolve()}")
    print("=" * 60)

    if not args.skip_generate:
        gen_argv = [
            "--outdir",
            str(args.outdir),
            "--resolution",
            RESOLUTION,
            "--n-samples",
            str(args.n_samples),
            "--seed",
            str(args.seed),
            "--cfg",
            str(args.cfg),
            "--tau",
            str(args.tau),
            "--start",
            str(args.start),
        ]
        if args.end is not None:
            gen_argv.extend(["--end", str(args.end)])
        if args.resume:
            gen_argv.append("--resume")
        if args.positive_prompt:
            gen_argv.append("--positive-prompt")

        print("\n[1/2] Generating images at 1024×1024 ...\n")
        run_module("geneval_generate", gen_argv)
    else:
        print("\n[1/2] Skipped generation (--skip-generate).\n")

    if not args.skip_eval:
        eval_argv = [
            str(args.outdir),
            "--outfile",
            str(results),
            "--detector-dir",
            str(args.detector_dir),
        ]
        if args.geneval_python:
            eval_argv.extend(["--python", args.geneval_python])

        print("\n[2/2] Running GenEval scoring ...\n")
        run_module("geneval_evaluate", eval_argv)
    else:
        print("\n[2/2] Skipped evaluation (--skip-eval).\n")

    print("\nDone.")
    if not args.skip_eval:
        print(f"Results: {results.resolve()}")


if __name__ == "__main__":
    main()
