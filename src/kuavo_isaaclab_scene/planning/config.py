"""Read collection YAML and enforce the output lane before creating artifacts."""

from __future__ import annotations

from pathlib import Path
import math
import re


OUTPUT_ROOT = Path("/home/work/mntvol/data/outputs")


def load_yaml(path: str | Path) -> dict:
    import yaml  # Optional planning dependency; do not affect existing launchers.

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with Path(path).open(encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return result


def output_directory(config: dict, run_name: str | None = None) -> Path:
    root = config.get("output_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("output_root must be an absolute path")
    if Path(root).resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"all outputs must be under {OUTPUT_ROOT}")
    name = run_name if run_name is not None else config.get("run_name")
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or name in {".", ".."}):
        raise ValueError("run_name must be one safe directory name, without slashes")
    result = Path(root) / name
    if result.is_symlink() or result.resolve().parent != OUTPUT_ROOT.resolve():
        raise ValueError("output run must not be a symlink or escape the output root")
    return result


def validate_collection(config: dict) -> None:
    """Strict production check; audit mode may inspect incomplete draft YAML."""
    if config.get("draft") is not False:
        raise ValueError("collection config is still a draft")
    for key in ("episodes", "max_attempts"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_attempts"] < config["episodes"]:
        raise ValueError("max_attempts cannot be below target episodes")
    fps = config.get("dataset_fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("dataset_fps must be finite and positive")
    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("specify head/left_wrist/right_wrist cameras explicitly")
    for name, camera in cameras.items():
        if not isinstance(camera, dict) or type(camera.get("enabled")) is not bool:
            raise ValueError(f"{name}.enabled must be boolean")
        if camera["enabled"]:
            for key in ("width", "height"):
                if type(camera.get(key)) is not int or camera[key] <= 0:
                    raise ValueError(f"{name}.{key} must be a positive integer")
    if config.get("target_format") != "lerobot_v3":
        raise ValueError("target_format must be lerobot_v3")
    if config.get("write_strategy") not in {"direct", "raw_then_convert"}:
        raise ValueError("select write_strategy after writer verification")
    for key in ("save_hdf5", "save_raw_control", "resume", "preserve_privileged_gt",
                "record_failure_metadata"):
        if type(config.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if config["write_strategy"] == "raw_then_convert" and not config["save_hdf5"]:
        raise ValueError("raw_then_convert requires save_hdf5")
    extras = config.get("extra_modalities")
    if (not isinstance(extras, dict) or set(extras) != {"depth", "segmentation"}
            or any(type(v) is not bool for v in extras.values())):
        raise ValueError("set depth/segmentation explicitly to booleans")
    output_directory(config)


def validate_randomization(config: dict) -> None:
    value = config.get("randomization")
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("randomization.enabled must be boolean")
    if not value["enabled"]:
        return

    def bounds(pair, name, nonnegative=False):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                or pair[0] > pair[1] or (nonnegative and pair[0] < 0)):
            raise ValueError(f"invalid [min,max] range: {name}")

    types = value.get("box_types")
    if (not isinstance(types, list) or not types or len(set(types)) != len(types)
            or not set(types) <= {"small", "medium", "large", "xlarge"}):
        raise ValueError("randomization.box_types must be unique supported box types")
    slots = value.get("rack_slot_ids")
    if not isinstance(slots, list) or not slots or any(not isinstance(s, str) or not s for s in slots):
        raise ValueError("select nonempty rack_slot_ids from the actual scene")
    position = value.get("box_position_offset_m")
    if not isinstance(position, dict) or position.get("frame") not in {"world", "rack"}:
        raise ValueError("box_position_offset_m.frame must be world or rack")
    for axis in ("x", "y", "z"):
        bounds(position.get(axis), axis)
    bounds(value.get("box_yaw_offset_rad"), "box_yaw_offset_rad")
    for name, key in (("lighting", "intensity_scale_range"), ("friction", "coefficient_range")):
        feature = value.get(name)
        if not isinstance(feature, dict) or type(feature.get("enabled")) is not bool:
            raise ValueError(f"randomization.{name}.enabled must be boolean")
        if feature["enabled"]:
            bounds(feature.get(key), f"{name}.{key}", nonnegative=True)
