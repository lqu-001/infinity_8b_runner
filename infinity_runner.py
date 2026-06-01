"""Shared Infinity-8B inference helpers for single-image and benchmark runners."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO = ROOT / "third_party" / "Infinity"
WEIGHTS_DIR = ROOT / "weights"
GENEVAL_REPO = ROOT / "third_party" / "geneval"
DEFAULT_GENEVAL_METADATA = GENEVAL_REPO / "prompts" / "evaluation_metadata.jsonl"


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def setup_pythonpath() -> None:
    repo = str(OFFICIAL_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def resolve_weights(resolution: str) -> tuple[Path, str]:
    if resolution == "512":
        return WEIGHTS_DIR / "infinity_8b_512x512_weights", "0.25M"
    return WEIGHTS_DIR / "infinity_8b_weights", "1M"


def validate_setup(resolution: str) -> tuple[Path, Path, Path]:
    ensure_exists(OFFICIAL_REPO / "tools" / "run_infinity.py", "Official Infinity repo not found. Run setup_official_infinity.ps1 first")
    transformer_dir, _ = resolve_weights(resolution)
    vae_path = WEIGHTS_DIR / "infinity_vae_d56_f8_14_patchify.pth"
    t5_path = WEIGHTS_DIR / "flan-t5-xl-official"
    ensure_exists(transformer_dir, "Missing Infinity-8B transformer weights. Run download_weights.py")
    ensure_exists(vae_path, "Missing Infinity VAE. Run download_weights.py")
    ensure_exists(t5_path, "Missing flan-t5-xl text encoder. Run download_weights.py")
    return transformer_dir, vae_path, t5_path


def build_infinity_args(
    resolution: str,
    *,
    cfg: str | float = "3",
    tau: float = 0.5,
    seed: int = 0,
    positive_prompt: bool = False,
) -> Namespace:
    transformer_dir, pn = resolve_weights(resolution)
    _, vae_path, t5_path = validate_setup(resolution)

    cfg_values = list(map(float, str(cfg).split(",")))
    cfg_parsed: float | list[float] = cfg_values[0] if len(cfg_values) == 1 else cfg_values

    return Namespace(
        cfg=cfg_parsed,
        tau=tau,
        pn=pn,
        model_path=str(transformer_dir),
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=str(vae_path),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_8b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=str(t5_path),
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        enable_positive_prompt=1 if positive_prompt else 0,
        cache_dir=str(ROOT / ".cache"),
        enable_model_cache=0,
        checkpoint_type="torch_shard",
        seed=seed,
        bf16=1,
    )


def load_pipeline(args: Namespace) -> dict[str, Any]:
    """Load tokenizer, VAE, Infinity, and scale schedule once for many generations."""
    setup_pythonpath()
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
    from tools.run_infinity import (
        gen_one_img,
        load_tokenizer,
        load_transformer,
        load_visual_tokenizer,
    )

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    return {
        "args": args,
        "text_tokenizer": text_tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "infinity": infinity,
        "scale_schedule": scale_schedule,
        "gen_one_img": gen_one_img,
    }


def generate_image(
    pipeline: dict[str, Any],
    prompt: str,
    output_path: Path,
    *,
    seed: int | None = None,
) -> Path:
    args = pipeline["args"]
    seed = args.seed if seed is None else seed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = pipeline["gen_one_img"](
        pipeline["infinity"],
        pipeline["vae"],
        pipeline["text_tokenizer"],
        pipeline["text_encoder"],
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=args.cfg,
        tau_list=args.tau,
        scale_schedule=pipeline["scale_schedule"],
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=args.enable_positive_prompt,
    )
    cv2.imwrite(str(output_path), generated.cpu().numpy())
    return output_path


def run_subprocess_generate(output: Path, args: SimpleNamespace) -> None:
    """Fallback: invoke official run_infinity.py in a subprocess (loads model each call)."""
    import subprocess

    infinity_args = build_infinity_args(
        args.resolution,
        cfg=args.cfg,
        tau=args.tau,
        seed=args.seed,
        positive_prompt=args.positive_prompt,
    )
    command = [
        sys.executable,
        str(OFFICIAL_REPO / "tools" / "run_infinity.py"),
        "--prompt",
        args.prompt,
        "--save_file",
        str(output),
        "--cfg",
        str(args.cfg),
        "--tau",
        str(args.tau),
        "--pn",
        infinity_args.pn,
        "--model_path",
        infinity_args.model_path,
        "--cfg_insertion_layer",
        str(infinity_args.cfg_insertion_layer),
        "--vae_type",
        str(infinity_args.vae_type),
        "--vae_path",
        infinity_args.vae_path,
        "--add_lvl_embeding_only_first_block",
        str(infinity_args.add_lvl_embeding_only_first_block),
        "--use_bit_label",
        str(infinity_args.use_bit_label),
        "--model_type",
        infinity_args.model_type,
        "--rope2d_each_sa_layer",
        str(infinity_args.rope2d_each_sa_layer),
        "--rope2d_normalized_by_hw",
        str(infinity_args.rope2d_normalized_by_hw),
        "--use_scale_schedule_embedding",
        str(infinity_args.use_scale_schedule_embedding),
        "--sampling_per_bits",
        str(infinity_args.sampling_per_bits),
        "--text_encoder_ckpt",
        infinity_args.text_encoder_ckpt,
        "--text_channels",
        str(infinity_args.text_channels),
        "--apply_spatial_patchify",
        str(infinity_args.apply_spatial_patchify),
        "--h_div_w_template",
        str(infinity_args.h_div_w_template),
        "--use_flex_attn",
        str(infinity_args.use_flex_attn),
        "--enable_positive_prompt",
        str(infinity_args.enable_positive_prompt),
        "--cache_dir",
        infinity_args.cache_dir,
        "--enable_model_cache",
        str(infinity_args.enable_model_cache),
        "--checkpoint_type",
        infinity_args.checkpoint_type,
        "--seed",
        str(args.seed),
        "--bf16",
        str(infinity_args.bf16),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OFFICIAL_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=OFFICIAL_REPO, env=env, check=True)
