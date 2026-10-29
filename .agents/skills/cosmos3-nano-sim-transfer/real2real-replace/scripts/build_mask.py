#!/usr/bin/env python3
"""Encode the SAM3 mask PNGs into controls/mask.mp4. Cosmos framework venv.

    python3 build_mask.py --exp EXP

The control hint is NOT built here. The framework derives edge and blur itself from
``vision_path``, at the geometry it resolves for the run, so computing one here would
only feed it a map detected at the wrong size. Seg and depth it cannot derive; those come
in as a precomputed video, passed to ``build_request.py --control-path``.

The mask is written at the source's own size and left for the framework to resize along
with everything else, so it goes through one resize instead of two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import probe, write_frames  # noqa: E402


def load_mask_frames(mask_dir: Path, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    paths = sorted(mask_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no mask PNGs in {mask_dir}")
    out = []
    for p in paths:
        im = Image.open(p).convert("L")
        if im.size != size:
            # NEAREST keeps the mask binary; a smooth resize would blur the boundary
            # into grey, and the pipeline thresholds at >16/255.
            im = im.resize(size, Image.NEAREST)
        out.append(np.array(im))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=Path)
    args = ap.parse_args()

    exp = args.exp
    src = exp / "inputs" / "source.mp4"
    mask_dir = exp / "inputs" / "mask_frames"
    info = probe(src)

    mask = load_mask_frames(mask_dir, (info.width, info.height))
    n = len(mask)
    mask_rgb = np.repeat(((mask > 127).astype(np.uint8) * 255)[..., None], 3, axis=3)
    mask_out = write_frames(mask_rgb, exp / "controls" / "mask.mp4", info.fps, lossless=True)

    generated = float((mask > 127).mean())
    summary = {
        "frames": int(n),
        "source_size": f"{info.width}x{info.height}",
        "mask_size": f"{info.width}x{info.height}",
        "fps": info.fps,
        "mask_path": str(mask_out),
        "generated_fraction": round(generated, 5),
        "pinned_fraction": round(1.0 - generated, 5),
    }
    (exp / "controls" / "mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if generated < 0.002:
        print("WARNING: the generated region is under 0.2% of pixels; the mask is probably wrong", file=sys.stderr)
    if generated > 0.97:
        print("WARNING: almost the whole frame is generated; pinning will constrain nothing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
