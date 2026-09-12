"""Encode Task1 JPEG capture sequences into an MP4 artifact."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def _ordered_frames(frame_dir: Path) -> list[Path]:
    frames = sorted(frame_dir.expanduser().resolve().glob("*.jpg"))
    if not frames:
        raise ValueError(f"No JPEG frames found in {frame_dir}")
    expected = [f"{index:05d}.jpg" for index in range(len(frames))]
    actual = [frame.name for frame in frames]
    if actual != expected:
        raise ValueError("Task1 video frames must be a contiguous zero-based JPEG sequence")
    return frames


def encode_jpeg_sequence(
    frame_dir: Path,
    video_path: Path,
    *,
    fps: float,
    overwrite: bool = False,
) -> dict[str, object]:
    """Encode captured JPEGs with ffmpeg, falling back to OpenCV MP4V."""
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    frames = _ordered_frames(frame_dir)
    output = video_path.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"Task1 video output must use the .mp4 extension: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Video already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-framerate",
                f"{fps:.8g}",
                "-i",
                str(frames[0].parent / "%05d.jpg"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        backend = "ffmpeg-libx264"
    else:
        import cv2

        first = cv2.imread(str(frames[0]))
        if first is None:
            raise ValueError(f"Could not read video frame: {frames[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not initialize the MP4V video writer")
        try:
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError(f"Invalid or inconsistent video frame: {frame_path}")
                writer.write(frame)
        except BaseException:
            writer.release()
            output.unlink(missing_ok=True)
            raise
        writer.release()
        backend = "opencv-mp4v"

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video encoder did not create a usable MP4: {output}")
    return {"backend": backend, "frame_count": len(frames)}
