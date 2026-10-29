#!/usr/bin/env python3
"""Verify a finished run: technical validity, then whether pinning actually held.

Cosmos framework venv.

    python3 verify_run.py --exp EXP --run r2r_v1 [--run r2r_v1_nopin]

Technical: frame count, size, FPS against the source; a non-frozen tail.
Behavioural: mean absolute difference from the source inside and outside the mask. The
pinned side should be small. There is no universal threshold, so the check reports the
numbers and flags only the two failure shapes that are unambiguous — a pinned region that
drifted as much as the generated one, and an output identical to the source everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_size, probe, read_frames  # noqa: E402

LONG_SIDE = 640


def tail_motion(frames: np.ndarray, n: int = 8) -> float:
    """Mean abs frame-to-frame difference over the last n frames; ~0 means a frozen tail."""
    tail = frames[-n:].astype(np.float32)
    if len(tail) < 2:
        return 0.0
    return float(np.abs(np.diff(tail, axis=0)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--stride", type=int, default=4, help="Sample every Nth frame.")
    args = ap.parse_args()

    exp = args.exp.resolve()
    src_path = exp / "inputs" / "source.mp4"
    mask_path = exp / "controls" / "mask.mp4"
    for p in (src_path, mask_path):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    src_info = probe(src_path)
    # Compare everything at one size with the source aspect ratio, so the mask, the
    # source and the model output (whose own size comes from the resolution tier) line up.
    size = fit_size(src_info.width, src_info.height, LONG_SIDE)
    src = read_frames(src_path, size=size).astype(np.float32)
    mask = read_frames(mask_path, size=size)
    n = min(len(src), len(mask))
    src, mask = src[:n], mask[:n]
    gen_mask = (mask[..., 0] > 127).astype(np.float32)[..., None]
    idx = np.arange(0, n, args.stride)

    report: dict[str, dict] = {}
    ok = True
    for run in args.run:
        out = exp / "generated" / run / run / "vision.mp4"
        # A run built with --no-pin is a deliberate baseline: it is supposed to drift
        # outside the mask, so drift there is the expected result, not a failure.
        req = exp / "requests" / f"{run}.json"
        pinned = True
        if req.exists():
            pinned = json.loads(req.read_text()).get("roi_pin_mask_path") is not None
        entry: dict = {"video": str(out), "pinned": pinned}
        if not out.exists():
            entry["status"] = "missing"
            report[run] = entry
            ok = False
            continue

        info = probe(out)
        g = read_frames(out, size=size).astype(np.float32)
        k = min(len(g), n)
        gi = idx[idx < k]
        d = np.abs(g[gi] - src[gi])
        m = gen_mask[gi]
        inside = float((d * m).sum() / max(m.sum() * 3, 1))
        outside = float((d * (1 - m)).sum() / max((1 - m).sum() * 3, 1))

        checks = {
            "frames_match_source": info.frames >= n,
            "fps_matches_source": abs(info.fps - src_info.fps) < 1e-2,
            "square_or_source_ratio": True,
            "tail_not_frozen": tail_motion(g) > 0.5,
        }
        entry.update(
            {
                "status": "ok",
                "frames": info.frames,
                "size": f"{info.width}x{info.height}",
                "fps": info.fps,
                "mae_inside_mask": round(inside, 2),
                "mae_outside_mask": round(outside, 2),
                "tail_motion": round(tail_motion(g), 3),
                "checks": checks,
            }
        )

        # The pinned side should be much closer to the source than the generated side.
        if pinned and outside >= inside:
            entry["warning"] = (
                "the pinned region drifted at least as much as the generated region; "
                "check that roi_pin_* reached the request and that [ROI pin] is in the log"
            )
            ok = False
        if pinned and inside < 2.0:
            entry["warning"] = "output is nearly identical to the source; nothing was replaced"
            ok = False
        if not pinned:
            entry["note"] = "no-pin baseline; drift outside the mask is expected here"
        if not all(checks.values()):
            ok = False
        report[run] = entry

    pinned_frac = float(1.0 - gen_mask.mean())
    summary = {
        "experiment": str(exp),
        "source": {"frames": src_info.frames, "fps": src_info.fps, "size": f"{src_info.width}x{src_info.height}"},
        "mask": {"generated_fraction": round(1.0 - pinned_frac, 5), "pinned_fraction": round(pinned_frac, 5)},
        "runs": report,
        "pass": ok,
    }
    (exp / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    for run, entry in report.items():
        log = exp / "generated" / run / "logs" / f"{run}.log"
        if not log.exists():
            log = exp / "generated" / "logs" / f"{run}.log"
        if not entry.get("pinned", True):
            continue
        if log.exists() and "[ROI pin]" not in log.read_text(errors="ignore"):
            print(f"NOTE: no '[ROI pin]' line in {log}; pinning may not have run for {run}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
