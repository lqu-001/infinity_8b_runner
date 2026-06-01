import argparse
from pathlib import Path
from types import SimpleNamespace

from infinity_runner import ROOT, build_infinity_args, generate_image, load_pipeline, run_subprocess_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Infinity-8B text-to-image inference.")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "infinity_8b.png")
    parser.add_argument("--resolution", choices=("512", "1024"), default="1024")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", default="3", help="Classifier-free guidance, e.g. 3 or comma-separated schedule.")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--positive-prompt", action="store_true", help="Enable official positive prompt augmentation.")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Invoke tools/run_infinity.py in a subprocess (reloads weights each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subprocess:
        run_subprocess_generate(
            args.output,
            SimpleNamespace(
                prompt=args.prompt,
                resolution=args.resolution,
                cfg=args.cfg,
                tau=args.tau,
                seed=args.seed,
                positive_prompt=args.positive_prompt,
            ),
        )
        return

    infinity_args = build_infinity_args(
        args.resolution,
        cfg=args.cfg,
        tau=args.tau,
        seed=args.seed,
        positive_prompt=args.positive_prompt,
    )
    pipeline = load_pipeline(infinity_args)
    generate_image(pipeline, args.prompt, args.output, seed=args.seed)
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
