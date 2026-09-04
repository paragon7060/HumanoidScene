#!/usr/bin/env python3
"""Generate final-size USD wrappers for the four articulated box templates."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "src" / "kuavo_isaaclab_scene" / "assets"
LAYOUT_MODULE = REPO_ROOT / "src" / "kuavo_isaaclab_scene" / "rack_box_layout.py"


def literal_constants(*names: str) -> dict[str, object]:
    """Read literal module constants without importing Isaac/Gym packages."""
    wanted = set(names)
    values: dict[str, object] = {}
    tree = ast.parse(LAYOUT_MODULE.read_text(encoding="utf-8"), filename=str(LAYOUT_MODULE))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id in wanted and node.value is not None:
            values[node.target.id] = ast.literal_eval(node.value)
    missing = wanted.difference(values)
    if missing:
        raise RuntimeError(f"Missing literal constants in {LAYOUT_MODULE}: {sorted(missing)}")
    return values


_CONSTANTS = literal_constants(
    "BOX_DIMENSIONS_M",
    "BOX_FLAP_LENGTH_M",
    "BOX_TEMPLATE_BODY_SIZE_M",
    "BOX_TYPE_LABELS",
)
BOX_DIMENSIONS_M = _CONSTANTS["BOX_DIMENSIONS_M"]
BOX_FLAP_LENGTH_M = _CONSTANTS["BOX_FLAP_LENGTH_M"]
BOX_TEMPLATE_BODY_SIZE_M = _CONSTANTS["BOX_TEMPLATE_BODY_SIZE_M"]
BOX_TYPE_LABELS = _CONSTANTS["BOX_TYPE_LABELS"]


def number(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def wrapper_text(box_type: str) -> str:
    asset_name = BOX_TYPE_LABELS[box_type]
    width, depth, body_height = BOX_DIMENSIONS_M[box_type]
    native_width, native_depth, native_body_height = BOX_TEMPLATE_BODY_SIZE_M
    root_scale = (
        width / native_width,
        depth / native_depth,
        body_height / native_body_height,
    )
    flap_length = BOX_FLAP_LENGTH_M[box_type]
    normalized_flap_length = flap_length / body_height
    flap_center_z = 1.005 + normalized_flap_length / 2.0

    def triple(values: tuple[float, float, float]) -> str:
        return "(" + ", ".join(number(value) for value in values) + ")"

    flap_blocks = []
    flap_specs = {
        "front": {
            "scale": (0.98, 0.01, normalized_flap_length),
            "translate": (0.0, 0.5, flap_center_z),
            "local_pos0": (0.0, 0.5, 1.005),
            "local_pos1": (0.0, 0.0, -0.5),
        },
        "back": {
            "scale": (0.98, 0.01, normalized_flap_length),
            "translate": (0.0, -0.5, flap_center_z),
            "local_pos0": (0.0, -0.5, 1.005),
            "local_pos1": (0.0, 0.0, -0.5),
        },
        "right": {
            "scale": (0.01, 0.98, normalized_flap_length),
            "translate": (0.5, 0.0, flap_center_z),
            "local_pos0": (0.5, 0.0, 1.005),
            "local_pos1": (0.0, 0.0, -0.5),
        },
        "left": {
            "scale": (0.01, 0.98, normalized_flap_length),
            "translate": (-0.5, 0.0, flap_center_z),
            "local_pos0": (-0.5, 0.0, 1.005),
            "local_pos1": (0.0, 0.0, -0.5),
        },
    }
    for flap_name, spec in flap_specs.items():
        flap_blocks.append(
            f'''        over Cube "flap_{flap_name}"
        {{
            double3 xformOp:scale = {triple(spec["scale"])}
            double3 xformOp:translate = {triple(spec["translate"])}

            over PhysicsRevoluteJoint "joint_{flap_name}"
            {{
                point3f physics:localPos0 = {triple(spec["local_pos0"])}
                point3f physics:localPos1 = {triple(spec["local_pos1"])}
            }}
        }}'''
        )

    return f'''#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
    customLayerData = {{
        string sourceTemplate = "{asset_name}.usd"
        string dimensions = "W={width:.3f}m D={depth:.3f}m bodyH={body_height:.3f}m flap={flap_length:.3f}m"
    }}
)

def Xform "Root" (
    prepend references = @./{asset_name}.usd@</Root>
)
{{
    over Xform "{asset_name}"
    {{
        float3 xformOp:scale = {triple(root_scale)}

{chr(10).join(flap_blocks)}
    }}
}}
'''


def main() -> None:
    for box_type, asset_name in BOX_TYPE_LABELS.items():
        output_path = ASSET_DIR / f"{asset_name}_physical.usda"
        output_path.write_text(wrapper_text(box_type), encoding="utf-8")
        print(f"wrote {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
