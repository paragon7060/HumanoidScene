"""Video exports retain evidence timing and work in browser MP4 players."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


def load_exporter():
    path = Path(__file__).resolve().parents[1]/'scripts/rl/browser_video.py'
    spec = importlib.util.spec_from_file_location('browser_video_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h264_export_preserves_frames_and_timing(tmp_path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    pytest.importorskip('imageio_ffmpeg')
    path = tmp_path/'evidence.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 2., (64,48))
    assert writer.isOpened()
    for value in (30,120,220):
        writer.write(np.full((48,64,3), value, dtype=np.uint8))
    writer.release()
    report = load_exporter().encode_browser_video(path)
    capture = cv2.VideoCapture(str(path))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == 2.
    # OpenCV can report the decoder's h264 FourCC instead of the MP4 avc1 tag.
    metadata = subprocess.run([load_exporter().ffmpeg_executable(), '-hide_banner',
        '-i', str(path)], capture_output=True, text=True).stderr
    stream = next(line for line in metadata.splitlines() if 'Video:' in line)
    assert 'h264' in stream and 'avc1' in stream and 'yuv420p' in stream
    for expected in (30,120,220):
        ok, frame = capture.read()
        assert ok and abs(float(frame.mean())-expected) < 10
    assert not capture.read()[0]
    capture.release()
    data = path.read_bytes()
    assert data.index(b'moov') < data.index(b'mdat')
    assert report['full_decode_verified']


def test_failed_export_preserves_original(tmp_path, monkeypatch):
    module = load_exporter()
    path = tmp_path/'evidence.mp4'
    path.write_bytes(b'original video')
    monkeypatch.setattr(module, 'ffmpeg_executable', lambda: 'ffmpeg')
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'ffmpeg')
    monkeypatch.setattr(module.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.encode_browser_video(path)
    assert path.read_bytes() == b'original video'
    assert list(tmp_path.iterdir()) == [path]
