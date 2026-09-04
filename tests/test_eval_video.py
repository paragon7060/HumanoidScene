from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")

from kuavo_isaaclab_scene.display.eval_video import (
    FfmpegVideoWriter,
    camera_mosaic_rgb,
    episode_video_path,
)


def test_episode_video_path_keeps_single_name_and_suffixes_multiple(tmp_path: Path) -> None:
    base = tmp_path / "evaluation.mp4"
    assert episode_video_path(base, 0, 1) == base.resolve()
    assert episode_video_path(base, 1, 3).name == "evaluation_ep001.mp4"
    with pytest.raises(ValueError, match=".mp4"):
        episode_video_path(tmp_path / "evaluation.avi", 0, 1)


def test_camera_mosaic_preserves_key_order_and_resizes_tiles() -> None:
    red = torch.zeros((1, 3, 2, 2))
    red[:, 0] = 1.0
    green = torch.zeros((1, 1, 1, 3), dtype=torch.uint8)
    green[..., 1] = 255
    blue = torch.zeros((3, 2, 2), dtype=torch.uint8)
    blue[2] = 255
    frame = camera_mosaic_rgb(
        {"head": red, "left": green, "right": blue},
        ("head", "left", "right"),
        output_height=2,
    )
    assert frame.dtype == torch.uint8
    assert frame.device.type == "cpu"
    assert frame.shape == (2, 6, 3)
    assert frame[0, 0].tolist() == [255, 0, 0]
    assert frame[0, 2].tolist() == [0, 255, 0]
    assert frame[0, 4].tolist() == [0, 0, 255]


def test_camera_mosaic_pads_odd_dimensions_for_h264() -> None:
    frame = camera_mosaic_rgb(
        {"camera": torch.ones((1, 3, 3, 3))},
        ("camera",),
    )
    assert frame.shape == (4, 4, 3)
    assert frame[3].sum().item() == 0
    assert frame[:, 3].sum().item() == 0


def test_camera_mosaic_rejects_invalid_output_height() -> None:
    with pytest.raises(ValueError, match="height must be positive"):
        camera_mosaic_rgb(
            {"camera": torch.ones((1, 3, 2, 2))},
            ("camera",),
            output_height=0,
        )


def test_video_writer_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "existing.mp4"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite-video"):
        FfmpegVideoWriter(path, width=4, height=4, fps=30.0)
    assert path.read_bytes() == b"keep"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_video_writer_creates_h264_mp4(tmp_path: Path) -> None:
    path = tmp_path / "two_frames.mp4"
    frame = torch.zeros((4, 6, 3), dtype=torch.uint8)
    frame[:, :, 0] = 255
    with FfmpegVideoWriter(path, width=6, height=4, fps=30.0) as writer:
        writer.write(frame)
        writer.write(frame)
    assert path.stat().st_size > 0
