#!/usr/bin/env python3
"""Assemble the 4-step + ROI-pin request JSON.

Every inference parameter below is fixed by the distilled checkpoint's recipe or by the
measured ablation. Only name, prompt, paths, seed and source-derived frame count / FPS /
aspect ratio change between runs.

The source goes in as ``vision_path`` and carries two jobs: the framework derives the
edge control from it at the geometry it resolves for the run, and the pinned region is
held at it.

    python3 build_request.py --exp EXP --name r2r_v1 --prompt-file EXP/prompts/plain.txt

Paths go in as they are. Only when the transfer step sees the filesystem differently --
a container, another host -- pass --repo-root (the local root) and --container-root (what
that side calls it) to have them rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALLOWED_ASPECTS, probe  # noqa: E402

# Baked into the Cosmos3-Nano-Sim-Transfer checkpoint; not free parameters.
# ``shift`` and ``sigma_max`` are deliberately absent: the distilled AR sampler takes its
# schedule from the checkpoint's own ``fixed_step_sampler_config.t_list`` and is never
# handed either one, so naming them here would only suggest they can be tuned.
FIXED = {
    "num_steps": 4,
    "guidance": 1.0,
    "control_guidance": 1.0,
    "normalize_cfg": False,
    "autoregressive": True,
    "num_video_frames_per_chunk": 81,
    "num_conditional_frames": 0,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "emphasize_control_in_prompt": False,
    "transfer_resize_mode": "direct_resize",
    "transfer_output_size": "model",
    "resolution": "480",
    "model_mode": "video2video",
    "negative_prompt": "",
    "duration_template": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
    "resolution_template": "This video is of {height}x{width} resolution.",
    "negative_metadata_mode": "none",
    "inverse_duration_template": "",
    "inverse_resolution_template": "",
    "negative_prompt_keep_metadata": False,
    "show_input": False,
    "show_control_condition": False,
    "video_save_quality": 10,
    "image_save_quality": 95,
    "enable_sound": False,
}


def container_path(local: Path, repo_root: Path, container_root: str | None) -> str:
    """Rewrite a local path to the path the transfer step will see.

    With no ``container_root`` the two sides share a filesystem view and the resolved
    local path goes in unchanged.
    """
    local = local.resolve()
    if container_root is None:
        return str(local)
    repo_root = repo_root.resolve()
    try:
        rel = local.relative_to(repo_root)
    except ValueError:
        raise SystemExit(
            f"{local} is outside {repo_root}; only --repo-root is mapped to "
            f"--container-root, so put the experiment under it, or widen --repo-root"
        )
    return f"{container_root.rstrip('/')}/{rel}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt-file", required=True, type=Path)
    ap.add_argument(
        "--control",
        default="edge",
        choices=["edge", "seg", "depth"],
        help="edge is derived by the framework from the source; seg and depth cannot be, so they need --control-path.",
    )
    ap.add_argument(
        "--edge-preset",
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="Canny preset the framework applies. medium (100/200) is the default and the "
        "measured best; read SKILL.md before lowering it to separate objects.",
    )
    ap.add_argument(
        "--control-path",
        type=Path,
        default=None,
        help="Precomputed control video. Required for --control seg or depth.",
    )
    ap.add_argument(
        "--pin-strength",
        type=float,
        default=1.0,
        help="1.0 unless an ablation calls for less. The measured sweep showed no "
        "inflection below 1.0 on a 90%%-pinned scene.",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--aspect-ratio",
        default=None,
        choices=[name for name, _ in ALLOWED_ASPECTS],
        help="Override the aspect ratio snapped from the source.",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Local root that --container-root maps to; defaults to the cwd. Only used when the runner puts the experiment under a different path, e.g. a container mount.",
    )
    ap.add_argument(
        "--container-root",
        default=None,
        help="What the transfer step calls --repo-root, e.g. a container mount. "
        "Omit when both sides see the same paths.",
    )
    ap.add_argument(
        "--no-pin",
        action="store_true",
        help="Paired control run: identical request with the roi_pin_* fields "
        "removed. Use it to prove pinning changed the output.",
    )
    args = ap.parse_args()

    exp = args.exp.resolve()
    src = exp / "inputs" / "source.mp4"
    mask = exp / "controls" / "mask.mp4"
    framework_derives_control = args.control == "edge"
    ctrl = None if framework_derives_control else args.control_path
    if ctrl is None and not framework_derives_control:
        print(f"--control {args.control} needs --control-path", file=sys.stderr)
        return 1
    for p in (src, mask, args.prompt_file) + ((ctrl,) if ctrl is not None else ()):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    info = probe(src)
    # The framework checks source/control alignment itself, but a mismatch caught here is
    # cheaper to read than the same failure several minutes into a GPU job.
    others = {"mask": probe(mask)}
    if ctrl is not None:
        others["control"] = probe(ctrl)
    for label, other in others.items():
        if abs(other.fps - info.fps) > 1e-3:
            print(f"FPS mismatch: source {info.fps}, {label} {other.fps}", file=sys.stderr)
            return 1
    n = min(info.frames, *(o.frames for o in others.values()))
    if n < 2:
        print(f"only {n} usable frames", file=sys.stderr)
        return 1

    prompt = args.prompt_file.read_text()
    if not prompt.strip():
        print("prompt file is empty", file=sys.stderr)
        return 1

    # aspect_ratio is a five-value enum in the request schema, not free dimensions.
    snapped, err = info.aspect_literal()
    aspect = args.aspect_ratio or snapped
    if args.aspect_ratio is None and err > 0.05:
        print(
            f"WARNING: source display ratio {info.dar or info.width / info.height:.3f} "
            f"snapped to {aspect} ({err * 100:.1f}% off); pass --aspect-ratio to choose",
            file=sys.stderr,
        )

    cp = lambda p: container_path(p, args.repo_root, args.container_root)  # noqa: E731
    req = dict(FIXED)
    req.update(
        {
            "name": args.name,
            "fps": float(info.fps),
            "num_frames": 81,
            "max_frames": int(n),
            "aspect_ratio": aspect,
            "prompt": prompt,
            "seed": args.seed,
            "vision_path": cp(src),
            args.control: (
                {"preset_edge_threshold": args.edge_preset} if framework_derives_control else {"control_path": cp(ctrl)}
            ),
        }
    )
    if not args.no_pin:
        req.update({"roi_pin_mask_path": cp(mask), "roi_pin_strength": args.pin_strength})

    out = exp / "requests" / f"{args.name}.json"
    out.write_text(json.dumps(req, indent=2) + "\n")
    print(
        json.dumps(
            {
                "request": str(out),
                "frames": int(n),
                "fps": info.fps,
                "aspect_ratio": req["aspect_ratio"],
                "source_size": f"{info.width}x{info.height}",
                "control": args.control,
                "control_source": "framework" if framework_derives_control else str(ctrl),
                "edge_preset": args.edge_preset if framework_derives_control else None,
                "pin": not args.no_pin,
                "pin_strength": None if args.no_pin else args.pin_strength,
                "prompt_chars": len(prompt),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
