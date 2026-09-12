from pathlib import Path

import cv2
from PIL import Image
import pytest


def test_jpeg_sequence_is_encoded_when_system_ffmpeg_is_missing(tmp_path, monkeypatch):
    try:
        from data_collection.task1 import video
    except ModuleNotFoundError as error:
        pytest.fail(f"Task1 video encoder is missing: {error}")

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (8, 6), color).save(frame_dir / f"{index:05d}.jpg")

    monkeypatch.setattr(video.shutil, "which", lambda _name: None)
    output = tmp_path / "result.mp4"
    result = video.encode_jpeg_sequence(frame_dir, output, fps=12.0)

    assert result == {"backend": "opencv-mp4v", "frame_count": 3}
    assert output.is_file() and output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 8
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 6
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0)
    finally:
        capture.release()
