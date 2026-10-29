#!/usr/bin/env python3
"""Stage a source clip: copy it in, decode PNG frames for SAM3, record metadata.

Runs under the Cosmos framework venv.

    python3 prep_source.py --video SRC.mp4 --exp-root OUTPUTS --name real2real_<tag>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_png_frames, experiment_dir, probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--exp-root", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Trim to this many frames; the whole pipeline then uses the trimmed clip.",
    )
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1

    exp = experiment_dir(args.exp_root, args.name)
    staged = exp / "inputs" / "source.mp4"
    shutil.copy2(args.video, staged)

    info = probe(staged)
    frames_dir = exp / "inputs" / "frames"
    dumped = dump_png_frames(staged, frames_dir, max_frames=args.max_frames)

    summary = {
        "experiment": str(exp),
        "source_original": str(args.video.resolve()),
        "source_staged": str(staged),
        "frames_dir": str(frames_dir),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frames_total": info.frames,
        "frames_used": dumped.frames,
    }
    (exp / "source_info.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
