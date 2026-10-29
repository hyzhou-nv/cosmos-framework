"""Shared helpers: experiment layout, frame IO, metadata.

Runs under the Cosmos framework venv. The SAM3 step runs under its own venv and
exchanges data with this module through PNG frame directories, because the SAM3
env has no video decoder.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# The only values the request schema accepts, as (literal, ratio).
ALLOWED_ASPECTS = [("1,1", 1.0), ("4,3", 4 / 3), ("3,4", 3 / 4), ("16,9", 16 / 9), ("9,16", 9 / 16)]


@dataclass
class MediaInfo:
    path: str
    frames: int
    width: int
    height: int
    fps: float
    dar: float = 0.0
    """Display aspect ratio. Differs from width/height when the pixels are not square."""

    def aspect_literal(self) -> tuple[str, float]:
        """Snap to the nearest allowed literal; also return the relative error."""
        ratio = self.dar or (self.width / self.height)
        name, best = min(ALLOWED_ASPECTS, key=lambda kv: abs(kv[1] - ratio))
        return name, abs(best - ratio) / ratio

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @staticmethod
    def load(path: Path) -> "MediaInfo":
        return MediaInfo(**json.loads(Path(path).read_text()))


def probe(video: Path) -> MediaInfo:
    """Read frame count, size and FPS without decoding the whole clip."""
    out = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    # ffmpeg's null muxer does not print the stream line, so parse the header dump.
    head = subprocess.run([ffmpeg_exe(), "-i", str(video)], capture_output=True, text=True).stderr
    width = height = 0
    fps = 0.0
    dar = 0.0
    for line in head.splitlines():
        if "Video:" not in line:
            continue
        # Non-square pixels: ffmpeg prints "736x544 [SAR 68:69 DAR 4:3]". The display
        # ratio is what the request's aspect_ratio must match, not the pixel dimensions.
        if "DAR " in line:
            tok = line.split("DAR ", 1)[1].split("]")[0].strip()
            a, _, b = tok.partition(":")
            try:
                dar = int(a) / int(b)
            except (ValueError, ZeroDivisionError):
                dar = 0.0
        for tok in line.split(","):
            tok = tok.strip()
            if "x" in tok and width == 0:
                a, _, b = tok.partition("x")
                if a.isdigit() and b.split()[0].isdigit():
                    width, height = int(a), int(b.split()[0])
            if tok.endswith("fps"):
                try:
                    fps = float(tok[:-3].strip())
                except ValueError:
                    pass
        break
    if width == 0 or height == 0 or fps == 0.0:
        raise RuntimeError(f"could not probe {video}:\n{head[-800:]}\n{out.stderr[-400:]}")
    raw = subprocess.run(
        [
            ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "scale=16:16",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    ).stdout
    frames = len(raw) // (16 * 16)
    return MediaInfo(str(video), frames, width, height, fps, dar)


def fit_size(width: int, height: int, longest: int) -> tuple[int, int]:
    """Scale to `longest` on the long side, keeping aspect; both sides stay even.

    Controls must carry the source's aspect ratio. The framework resizes them again with
    the request's own `aspect_ratio`, so a control squashed to square would be stretched
    back and no longer line up with the footage.
    """
    if width >= height:
        w = longest
        h = max(2, round(longest * height / width))
    else:
        h = longest
        w = max(2, round(longest * width / height))
    return w - (w % 2), h - (h % 2)


def read_frames(
    video: Path,
    size: tuple[int, int] | int | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    """Decode a clip to uint8 [T, H, W, 3]. `size` is (w, h), or an int for a square."""
    info = probe(video)
    if size is None:
        w, h = info.width, info.height
        vf = "null"
    else:
        w, h = (size, size) if isinstance(size, int) else size
        vf = f"scale={w}:{h}:flags=bicubic"
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)
    return arr[:max_frames] if max_frames else arr


def write_frames(frames: np.ndarray, out: Path, fps: float, lossless: bool = False) -> Path:
    """Encode uint8 [T, H, W, 3] to mp4. Control and mask videos use lossless."""
    import imageio_ffmpeg

    out.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    params = ["-crf", "0"] if lossless else ["-crf", "14"]
    writer = imageio_ffmpeg.write_frames(
        str(out),
        (w, h),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv444p" if lossless else "yuv420p",
        output_params=params,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    return out


def dump_png_frames(video: Path, out_dir: Path, max_frames: int | None = None) -> MediaInfo:
    """Decode to numbered PNGs so the SAM3 venv, which has no decoder, can read them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-start_number", "0", str(out_dir / "%06d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    info = probe(video)
    if max_frames:
        info.frames = min(info.frames, max_frames)
    info.dump(out_dir / "meta.json")
    return info


def load_png_frames(frame_dir: Path) -> np.ndarray:
    from PIL import Image

    paths = sorted(Path(frame_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frame_dir}")
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])


def experiment_dir(root: Path, name: str) -> Path:
    """One unique directory per attempt; never write into an existing one."""
    d = Path(root) / name
    if d.exists():
        raise FileExistsError(f"{d} already exists; pick a new experiment name")
    for sub in ("inputs", "controls", "requests", "prompts", "generated", "logs"):
        (d / sub).mkdir(parents=True)
    return d
