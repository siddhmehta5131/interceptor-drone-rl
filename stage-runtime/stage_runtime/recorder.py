"""Video recording: one MP4 per stage (imageio-ffmpeg), with GIF/PNG fallbacks.

MP4 is the default and needs ``imageio-ffmpeg`` for its bundled ffmpeg binary.
``matplotlib.animation.FFMpegWriter`` is used when that backend is available;
otherwise we fall back to Pillow (GIF) or a numbered PNG sequence so a render
never dies just because a codec is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation      # noqa: E402


class RecorderError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    """True when matplotlib can write MP4/MOV through ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    try:
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:            # pragma: no cover - depends on wheel layout
        return False
    return matplotlib.animation.writers.is_available("ffmpeg")


def available_formats() -> tuple:
    out = ["png"]
    if ffmpeg_available():
        out.insert(0, "mp4")
    try:
        import PIL  # noqa: F401
        out.insert(1 if out[0] == "mp4" else 0, "gif")
    except ImportError:
        pass
    return tuple(out)


class FrameRecorder:
    """Append-only frame sink writing one file for a whole stage.

    ``fps`` is fixed at construction; frames are grabbed from a live figure so
    the caller keeps full control over what is drawn.
    """

    def __init__(self, path: Path, *, fmt: str = "mp4", fps: int = 20,
                 dpi: int = 100, bitrate: int = 4000):
        self.path = Path(path)
        self.fmt = fmt
        self.fps = int(fps)
        self.dpi = int(dpi)
        self.bitrate = int(bitrate)
        self.count = 0
        self._writer = None
        self._fig = None
        self._suffix = ".mp4" if fmt == "mp4" else f".{fmt}"
        self._open()

    # -- lifecycle ------------------------------------------------------

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == "mp4":
            if not ffmpeg_available():
                raise RecorderError(
                    "MP4 output needs imageio-ffmpeg: pip install imageio imageio-ffmpeg "
                    "(the stage Docker image already includes it)"
                )
            self._writer = matplotlib.animation.FFMpegWriter(
                fps=self.fps,
                codec="libx264",
                bitrate=self.bitrate,
                metadata={"title": self.path.stem, "artist": "interceptor-stage-runtime"},
            )
            self._is_setup = False
        elif self.fmt == "gif":
            self._writer = matplotlib.animation.PillowWriter(fps=self.fps)
            self._is_setup = False
        elif self.fmt == "png":
            self._writer = None      # numbered files, written in `grab`
        else:
            raise RecorderError(f"unsupported video format {self.fmt!r}")

    def grab(self, fig: "plt.Figure") -> None:
        """Grab the current canvas of ``fig``."""
        if self.fmt == "png":
            out = self.path.with_name(f"{self.path.stem}_{self.count:05d}.png")
            fig.savefig(out, dpi=self.dpi)
        else:
            if not getattr(self, "_is_setup", True):
                self._writer.setup(fig, str(self.path), dpi=self.dpi)
                self._is_setup = True
            self._writer.grab_frame()
        self.count += 1

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.finish()
            except Exception as exc:     # pragma: no cover
                raise RecorderError(f"failed to finish {self.path}: {exc}") from exc
            finally:
                self._writer = None

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "FrameRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        return False

    # -- info -------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "format": self.fmt,
            "fps": self.fps,
            "frames": self.count,
            "seconds": round(self.count / float(self.fps), 3) if self.fps else 0.0,
            "bytes": self.path.stat().st_size if self.path.is_file() else 0,
        }


def probe_video(path: Path) -> dict:
    """Container/codec facts via ffprobe when present, else file size only."""
    import shutil
    import subprocess

    out = {"path": str(path), "exists": Path(path).is_file()}
    if not out["exists"]:
        return out
    out["bytes"] = Path(path).stat().st_size
    exe = shutil.which("ffprobe")
    if exe:
        try:
            res = subprocess.run(
                [exe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames",
                 "-show_entries", "stream=codec_name,width,height,nb_read_frames",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=60, check=False,
            )
            import json
            payload = json.loads(res.stdout or "{}")
            stream = (payload.get("streams") or [{}])[0]
            out.update({
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "frames": int(stream["nb_read_frames"]) if stream.get("nb_read_frames") else None,
            })
        except Exception:                # pragma: no cover - diagnostics only
            pass
    return out


def ffmpeg_exe() -> Optional[str]:
    """Path to the ffmpeg binary matplotlib will use (if any)."""
    if not ffmpeg_available():
        return None
    return matplotlib.animation.ffmpeg