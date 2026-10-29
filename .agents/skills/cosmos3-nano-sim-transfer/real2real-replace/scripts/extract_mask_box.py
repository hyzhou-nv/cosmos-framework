#!/usr/bin/env python3
"""Text-prompted SAM3 tracking narrowed to a user bbox, with a backward pass.

Why this exists: the video model's session API takes text prompts only, so a bbox
cannot seed the tracker directly. Instead we track the text noun, keep only the
object that lands inside the bbox, and fill leading frames the detector missed by
propagating backward from the first frame the object was seen on.

    --prompt cube --box 408,250,470,305 --frames-dir ... --out-dir ...
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


def load_frames(d: Path) -> np.ndarray:
    paths = sorted(d.glob("*.png"))
    if not paths:
        raise FileNotFoundError(d)
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[:, None]
    t = torch.nn.functional.max_pool2d(t, 2 * r + 1, stride=1, padding=r)
    return t[:, 0].numpy() > 0.5


def collect(model, processor, session, start, reverse, t, h, w, per_obj, min_score):
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, start_frame_idx=start, reverse=reverse):
            res = processor.postprocess_outputs(session, out)
            i = out.frame_idx
            if i is None or i >= t:
                continue
            masks, scores, ids = res["masks"], res["scores"], res["object_ids"]
            if masks is None or len(masks) == 0:
                continue
            keep = scores.float().cpu().numpy() >= min_score
            if not keep.any():
                continue
            m = masks.cpu().numpy().astype(bool)[keep]
            kept = [int(o) for o in np.asarray(ids.cpu() if hasattr(ids, "cpu") else ids)[keep]]
            for oid, mi in zip(kept, m):
                per_obj.setdefault(oid, np.zeros((t, h, w), bool))[i] |= mi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument(
        "--box",
        action="append",
        required=True,
        help="x0,y0,x1,y1 in source pixels; keeps only objects centred inside it.",
    )
    ap.add_argument("--target", choices=["foreground", "background"], default="foreground")
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        required=DEFAULT_MODEL is None,
        help="Local SAM3 weights directory; defaults to $SAM3_MODEL",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    from transformers import Sam3VideoModel, Sam3VideoProcessor

    boxes = [tuple(float(v) for v in b.split(",")) for b in args.box]

    frames = load_frames(args.frames_dir)
    t, h, w, _ = frames.shape
    print(f"frames {t} @ {w}x{h}", flush=True)

    dtype = getattr(torch, args.dtype)
    processor = Sam3VideoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = (
        Sam3VideoModel.from_pretrained(str(args.model), local_files_only=True, attn_implementation="sdpa", dtype=dtype)
        .cuda()
        .eval()
    )
    session = processor.init_video_session(video=list(frames), inference_device="cuda", dtype=dtype)
    processor.add_text_prompt(session, list(args.prompt))

    per_obj: dict[int, np.ndarray] = {}
    collect(model, processor, session, 0, False, t, h, w, per_obj, args.min_score)
    print(f"forward pass tracked {len(per_obj)} objects", flush=True)

    # Keep objects whose centroid sits inside one of the boxes on their first frame.
    chosen: dict[int, np.ndarray] = {}
    for oid, m in per_obj.items():
        nz = [i for i in range(t) if m[i].any()]
        if not nz:
            continue
        ys, xs = np.where(m[nz[0]])
        cx, cy = xs.mean(), ys.mean()
        for x0, y0, x1, y1 in boxes:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                chosen[oid] = m
                print(f"  keep obj {oid}: first frame {nz[0]}, centroid ({cx:.0f},{cy:.0f})", flush=True)
                break
    if not chosen:
        print("no tracked object fell inside the box", file=sys.stderr)
        return 1

    first = min(i for m in chosen.values() for i in range(t) if m[i].any())
    if first > 0:
        print(f"first detection at frame {first}; running a backward pass to fill 0..{first - 1}", flush=True)
        back: dict[int, np.ndarray] = {}
        collect(model, processor, session, first, True, t, h, w, back, args.min_score)
        for oid in list(chosen):
            if oid in back:
                chosen[oid] |= back[oid]

    obj = np.logical_or.reduce(list(chosen.values()))
    still = [i for i in range(t) if not obj[i].any()]
    obj = dilate(obj, args.dilate)
    generate = obj if args.target == "foreground" else ~obj

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.png"):
        stale.unlink()
    for i, g in enumerate(generate):
        Image.fromarray((g.astype(np.uint8) * 255)).save(args.out_dir / f"{i:06d}.png")

    empty = [i for i, g in enumerate(generate) if not g.any()]
    summary = {
        "frames": t,
        "width": w,
        "height": h,
        "prompts": args.prompt,
        "boxes": args.box,
        "target": args.target,
        "dilate_px": args.dilate,
        "min_score": args.min_score,
        "objects_kept": sorted(chosen),
        "first_detection_frame": first,
        "frames_still_empty_after_backward": still,
        "object_coverage_dilated": round(float(obj.mean()), 5),
        "generated_fraction": round(float(generate.mean()), 5),
        "pinned_fraction": round(1.0 - float(generate.mean()), 5),
        "frames_with_empty_mask": empty,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir.parent / "mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if empty:
        print(f"WARNING: {len(empty)} frames still have an all-black mask", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
