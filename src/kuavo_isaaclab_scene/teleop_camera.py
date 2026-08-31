"""Read single-environment RTX frames without Isaac Lab's fixed-size cache."""

import numpy as np


def _annotator_array(camera, name):
    # Lab 2.3's Camera.data copies into a preallocated tensor before the caller
    # can check it. A transient empty frame on XR reconnect therefore raises
    # a shape error there. Check raw annotator output before conversion/copy.
    output = camera._rep_registry[name][0].get_data()
    raw = output.get("data") if isinstance(output, dict) else output
    if raw is None or not getattr(raw, "shape", ()) or 0 in raw.shape:
        return None
    data, _ = camera._process_annotator_output(name, output)
    return data.detach().cpu().numpy()


def camera_rgb(camera):
    rgb = _annotator_array(camera, "rgb")
    if rgb is None or rgb.ndim != 3 or rgb.shape[:2] != tuple(camera.image_shape):
        return None
    if rgb.shape[-1] not in (3, 4):
        return None
    rgb = rgb[..., :3]
    if rgb.dtype.kind == "f":
        if not np.isfinite(rgb).all():
            return None
        rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.01 else 1.0), 0, 255)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def camera_depth(camera):
    depth = _annotator_array(camera, "distance_to_image_plane")
    if depth is None or depth.shape[:2] != tuple(camera.image_shape):
        return None
    return depth.astype(np.float16)
