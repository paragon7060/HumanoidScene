"""Finalize local experiment videos as browser-compatible H.264 MP4.

Call only after the source writer has stopped. Original experiment archives can
remain immutable by passing a separate destination when repairing old media.
Encoding and verification run on CPU, without an Isaac runtime or GPU.
"""
from pathlib import Path
import shutil
import subprocess
import uuid


def ffmpeg_executable():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('H.264 export requires imageio-ffmpeg or ffmpeg')
        return executable


def encode_browser_video(source, destination=None):
    """Encode, fully decode-check, then atomically publish the completed MP4.

    Preserve the source timestamps, frame count and dimensions. No interpolation
    or physics replay is performed. The original survives any encoding failure.
    """
    source = Path(source)
    destination = Path(destination) if destination is not None else source
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError('Source video is missing or empty')
    if destination != source and destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f'.{destination.stem}.{uuid.uuid4().hex}.pending.mp4')
    executable = ffmpeg_executable()
    try:
        subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-i', str(source), '-map', '0:v:0', '-an', '-c:v', 'libx264',
            '-threads', '1', '-preset', 'fast', '-crf', '18', '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p', '-tag:v', 'avc1', '-fps_mode', 'passthrough',
            '-movflags', '+faststart', str(pending)], check=True)
        subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-xerror', '-i', str(pending), '-map', '0:v:0', '-f', 'null', '-'], check=True)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return {'codec': 'h264', 'tag': 'avc1', 'pixel_format': 'yuv420p',
            'faststart': True, 'full_decode_verified': True}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(encode_browser_video(args.source, args.destination)))
