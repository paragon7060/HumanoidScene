"""Headless-safe camera mosaic recording for policy evaluation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional


def episode_video_path(base_path: Path, episode_index: int, episode_count: int) -> Path:
    """Return one stable output path per episode without changing single-run names."""
    path = base_path.expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Evaluation video output must use the .mp4 extension: {path}")
    if episode_count <= 0:
        raise ValueError("Episode count must be positive.")
    if not 0 <= episode_index < episode_count:
        raise ValueError(
            f"Episode index {episode_index} is outside the configured count {episode_count}."
        )
    if episode_count == 1:
        return path
    return path.with_name(f"{path.stem}_ep{episode_index:03d}{path.suffix}")


def _camera_chw(value: Any, key: str) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Video recording supports one environment; {key} has batch {tensor.shape[0]}."
            )
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Camera {key} must be CHW or HWC; received {tuple(tensor.shape)}.")
    if tensor.shape[0] in (3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"Camera {key} must contain three RGB channels.")
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
        if tensor.numel() and float(tensor.detach().amax().item()) > 1.5:
            tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def camera_mosaic_rgb(
    observation: Mapping[str, Any],
    camera_keys: Sequence[str],
    *,
    output_height: int | None = None,
) -> torch.Tensor:
    """Compose policy cameras left-to-right into an even-sized CPU RGB frame."""
    if not camera_keys:
        raise ValueError("At least one camera key is required for video recording.")
    cameras = []
    for key in camera_keys:
        if key not in observation:
            raise KeyError(f"Video camera key is missing from the policy observation: {key}")
        cameras.append(_camera_chw(observation[key], key))
    if any(camera.shape[1] == 0 or camera.shape[2] == 0 for camera in cameras):
        raise ValueError("Video cameras must have non-empty height and width dimensions.")
    target_height = (
        output_height
        if output_height is not None
        else max(int(camera.shape[1]) for camera in cameras)
    )
    if target_height <= 0:
        raise ValueError("Video height must be positive.")

    resized = []
    for camera in cameras:
        height, width = int(camera.shape[1]), int(camera.shape[2])
        target_width = max(1, round(width * target_height / height))
        if (height, width) != (target_height, target_width):
            camera = torch_functional.interpolate(
                camera.unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        resized.append(camera)
    mosaic = torch.cat(resized, dim=2)
    frame = (
        mosaic.mul(255.0)
        .round_()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
    )
    padded_height = int(frame.shape[0]) + int(frame.shape[0] % 2)
    padded_width = int(frame.shape[1]) + int(frame.shape[1] % 2)
    if (padded_height, padded_width) != tuple(frame.shape[:2]):
        padded = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8)
        padded[: frame.shape[0], : frame.shape[1]] = frame
        frame = padded
    return frame


class FfmpegVideoWriter:
    """Stream RGB frames to a portable H.264 MP4 using the system ffmpeg."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        overwrite: bool = False,
        ffmpeg_binary: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("H.264 video width and height must be positive even numbers.")
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        self.path = path.expanduser().resolve()
        if self.path.suffix.lower() != ".mp4":
            raise ValueError(f"Evaluation video output must use the .mp4 extension: {self.path}")
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"Video already exists: {self.path}. Pass --overwrite-video to replace it."
            )
        ffmpeg = ffmpeg_binary or shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg is required for --video-out. Install it with your OS package manager."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self._closed = False
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.8g}",
            "-i",
            "-",
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
            str(self.path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed video writer.")
        if (
            frame.dtype != torch.uint8
            or frame.device.type != "cpu"
            or tuple(frame.shape) != (self.height, self.width, 3)
        ):
            raise ValueError(
                "Video frame must be a CPU uint8 tensor with shape "
                f"({self.height}, {self.width}, 3); received {frame.device}, "
                f"{frame.dtype}, {tuple(frame.shape)}."
            )
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable.")
        try:
            self._process.stdin.write(frame.numpy().tobytes())
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RuntimeError(f"ffmpeg stopped while writing {self.path}: {error}") from exc

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "no diagnostic output"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pipe_error: BrokenPipeError | None = None
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError as exc:
                pipe_error = exc
        return_code = self._process.wait()
        error = self._read_error()
        if return_code != 0 or pipe_error is not None:
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code} while writing {self.path}: {error}"
            ) from pipe_error

    def __enter__(self) -> "FfmpegVideoWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
