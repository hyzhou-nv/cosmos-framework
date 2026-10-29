#!/usr/bin/env python3
"""Text-prompted SAM3 video segmentation -> a binary mask the pin pipeline can read.

Runs under the SAM3 venv, which has no video decoder, so it reads the PNG frames
written by prep_source.py and writes mask PNGs back. build_mask.py (Cosmos venv)
encodes them.

Mask polarity follows the pin pipeline: WHITE is generated, BLACK is pinned to the
source. `--target foreground` makes the matched objects white; `--target background`
inverts that, so the objects stay and everything else is regenerated.

    python3 extract_mask.py --frames-dir EXP/inputs/frames --out-dir EXP/inputs/mask_frames \
        --prompt "car" --prompt "person" --target foreground --dilate 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Local SAM3 weights. Set SAM3_MODEL, or pass --model; there is no site default.
DEFAULT_MODEL = Path(os.environ["SAM3_MODEL"]) if os.environ.get("SAM3_MODEL") else None


def load_frames(frame_dir: Path) -> np.ndarray:
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frame_dir}")
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])


def merge_duplicates(per_obj: dict[int, np.ndarray], containment: float = 0.5) -> list[list[int]]:
    """Group object ids that cover the same physical thing.

    Two ids are duplicates when most of the smaller one sits inside the larger, which is
    what happens when several text prompts match one object. Returns one list of ids per
    distinct object.
    """
    ids = sorted(per_obj)
    areas = {o: int(per_obj[o].sum()) for o in ids}
    parent = {o: o for o in ids}

    def find(o: int) -> int:
        while parent[o] != o:
            parent[o] = parent[parent[o]]
            o = parent[o]
        return o

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            smaller = min(areas[a], areas[b])
            if smaller == 0:
                continue
            inter = int(np.logical_and(per_obj[a], per_obj[b]).sum())
            if inter / smaller >= containment:
                parent[find(a)] = find(b)

    groups: dict[int, list[int]] = {}
    for o in ids:
        groups.setdefault(find(o), []).append(o)
    return list(groups.values())


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element dilation via max pooling; no cv2 in this env."""
    if radius <= 0:
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[:, None]
    k = 2 * radius + 1
    t = torch.nn.functional.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return t[:, 0].numpy() > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Noun phrase for one class; repeat for several. SAM3 is text-grounded, "
        "so use concrete nouns ('parked car', 'person'), not instructions.",
    )
    ap.add_argument("--target", choices=["foreground", "background"], default="foreground")
    ap.add_argument(
        "--dilate",
        type=int,
        default=4,
        help="Grow the object mask by this many pixels before use, to cover "
        "contact shadows and matting error at object edges.",
    )
    ap.add_argument(
        "--keep-instances-apart",
        action="store_true",
        help="Dilate each tracked object on its own and keep a one-pixel gap "
        "where two of them would touch, so adjacent objects stay separate "
        "regions instead of merging into one blob.",
    )
    ap.add_argument(
        "--save-instances",
        action="store_true",
        help="Also write per-object mask PNGs under <out-dir>/instances/ and "
        "report each object's coverage. Diagnostic only; the pipeline "
        "reads the combined mask.",
    )
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        required=DEFAULT_MODEL is None,
        help="Local SAM3 weights directory; defaults to $SAM3_MODEL",
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model and session dtype. They must match, or the memory encoder's conv sees mixed input and bias types.",
    )
    args = ap.parse_args()

    from transformers import Sam3VideoModel, Sam3VideoProcessor

    frames = load_frames(args.frames_dir)
    if args.max_frames:
        frames = frames[: args.max_frames]
    t, h, w, _ = frames.shape
    print(f"frames {t} @ {w}x{h}", flush=True)

    dtype = getattr(torch, args.dtype)
    processor = Sam3VideoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = (
        Sam3VideoModel.from_pretrained(
            str(args.model),
            local_files_only=True,
            attn_implementation="sdpa",
            dtype=dtype,
        )
        .cuda()
        .eval()
    )
    print(f"SAM3 loaded ({args.dtype})", flush=True)

    session = processor.init_video_session(video=list(frames), inference_device="cuda", dtype=dtype)
    processor.add_text_prompt(session, list(args.prompt))

    obj_mask = np.zeros((t, h, w), dtype=bool)
    # Per object id, so adjacent objects can be grown without being fused.
    per_obj: dict[int, np.ndarray] = {}
    per_prompt_ids: dict[str, set[int]] = {p: set() for p in args.prompt}
    seen = 0

    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
            res = processor.postprocess_outputs(session, out)
            idx = out.frame_idx
            if idx is None or idx >= t:
                continue
            seen += 1
            masks = res["masks"]
            scores = res["scores"]
            ids = res["object_ids"]
            for prompt, oids in (res.get("prompt_to_obj_ids") or {}).items():
                if prompt in per_prompt_ids:
                    per_prompt_ids[prompt].update(int(o) for o in oids)
            if masks is None or len(masks) == 0:
                continue
            keep = scores.float().cpu().numpy() >= args.min_score
            if not keep.any():
                continue
            m = masks.cpu().numpy().astype(bool)[keep]
            obj_mask[idx] |= m.any(axis=0)
            kept_ids = [int(o) for o in np.asarray(ids.cpu() if hasattr(ids, "cpu") else ids)[keep]]
            for oid, mi in zip(kept_ids, m):
                per_obj.setdefault(oid, np.zeros((t, h, w), dtype=bool))[idx] |= mi

    if seen != t:
        print(f"WARNING: tracked {seen} frames but the clip has {t}", file=sys.stderr)

    coverage_raw = float(obj_mask.mean())
    touching = 0
    merged_groups: list[list[int]] = []
    if args.keep_instances_apart and len(per_obj) > 1:
        # Two prompts routinely match the same physical object ("towel" and "folded
        # clothing" over one stack), and those duplicates overlap almost completely.
        # Merge them first, or the seam step below deletes their shared interior and
        # guts the mask. Only genuinely disjoint objects get a seam between them.
        merged_groups = merge_duplicates(per_obj)
        merged = [np.logical_or.reduce([per_obj[o] for o in g]) for g in merged_groups]
        grown = np.stack([dilate(m, args.dilate) for m in merged])
        overlap = grown.sum(axis=0) > 1
        touching = int(overlap.sum())
        obj_mask = grown.any(axis=0) & ~overlap
    else:
        obj_mask = dilate(obj_mask, args.dilate)
    coverage_dilated = float(obj_mask.mean())

    generate = obj_mask if args.target == "foreground" else ~obj_mask

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.png"):
        stale.unlink()
    for i, g in enumerate(generate):
        Image.fromarray((g.astype(np.uint8) * 255)).save(args.out_dir / f"{i:06d}.png")

    if args.save_instances and per_obj:
        inst_dir = args.out_dir / "instances"
        inst_dir.mkdir(parents=True, exist_ok=True)
        for stale in inst_dir.glob("*.png"):
            stale.unlink()
        for oid, m in sorted(per_obj.items()):
            for i in (0, t // 2, t - 1):
                Image.fromarray(m[i].astype(np.uint8) * 255).save(inst_dir / f"obj{oid:03d}_f{i:06d}.png")

    empty = [i for i, g in enumerate(generate) if not g.any()]
    summary = {
        "frames": int(t),
        "width": int(w),
        "height": int(h),
        "prompts": list(args.prompt),
        "target": args.target,
        "dilate_px": args.dilate,
        "keep_instances_apart": args.keep_instances_apart,
        "instances_tracked": len(per_obj),
        "distinct_objects": len(merged_groups) or len(per_obj),
        "duplicate_groups": [g for g in merged_groups if len(g) > 1],
        "instance_coverage": {str(o): round(float(m.mean()), 5) for o, m in sorted(per_obj.items())},
        "seam_pixels_removed": touching,
        "min_score": args.min_score,
        "object_coverage_raw": round(coverage_raw, 5),
        "object_coverage_dilated": round(coverage_dilated, 5),
        "generated_fraction": round(float(generate.mean()), 5),
        "pinned_fraction": round(1.0 - float(generate.mean()), 5),
        "objects_per_prompt": {p: len(v) for p, v in per_prompt_ids.items()},
        "frames_with_empty_mask": empty,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir.parent / "mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    for prompt, oids in per_prompt_ids.items():
        if not oids:
            print(f"WARNING: prompt {prompt!r} matched nothing", file=sys.stderr)
    if empty:
        print(f"WARNING: {len(empty)} frames have an all-black mask", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
