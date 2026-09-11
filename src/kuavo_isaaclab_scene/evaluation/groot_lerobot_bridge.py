"""LeRobot/GR00T bridge for the Kuavo ManagerBased environment.

This module keeps policy I/O separate from the Isaac Lab application launcher so
its action conversion and image formatting can be unit-tested without starting
Isaac Sim.  The public LeRobot keys intentionally match the dataset convention:
``observation.state``, ``observation.images.<camera>``, ``task``, and ``action``.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

from ..core.paths import PACKAGE_IMPORT_ROOT
from ..robots.gripper_io import claw_fraction, resolve_gripper_views
from .policy_profiles import (
    ARM_CLAW_PROFILES,
    DEFAULT_POLICY_PROFILE,
    POLICY_PROFILES,
    RWH_KUAVO_V2_S56_PROFILE,
)


CONTROLLED_JOINT_NAMES = (
    "waist_yaw_joint",
    *(f"zarm_l{index}_joint" for index in range(1, 8)),
    *(f"zarm_r{index}_joint" for index in range(1, 8)),
)
MANAGER_ACTION_SCALES = (1.0, *(0.45 for _ in range(14)))
STATE_KEY = "observation.state"
ACTION_KEY = "action"
RWH_KUAVO_V2_NAMES = (
    *(f"zarm_l{index}" for index in range(1, 8)),
    "left_claw",
    *(f"zarm_r{index}" for index in range(1, 8)),
    "right_claw",
)
DEFAULT_CAMERA_MAP = {
    "observation.images.head": "robustness_camera",
    "observation.images.waist": "waist_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
}
RWH_KUAVO_V2_CAMERA_MAP = {
    "observation.images.head_cam_h": "robustness_camera",
    "observation.images.wrist_cam_l": "left_wrist_camera",
    "observation.images.wrist_cam_r": "right_wrist_camera",
}
ACTION_MODES = ("manager", "joint_position", "joint_delta")
STATE_MODES = ("manager", "joint_position")


@dataclass(frozen=True)
class ActionAdaptation:
    """Converted Isaac Lab action and clipping diagnostics."""

    action: torch.Tensor
    unclipped_action: torch.Tensor
    saturation_fraction: float


@dataclass(frozen=True)
class InferenceSample:
    """One action returned by a policy and whether a new chunk was inferred."""

    action: torch.Tensor
    inferred_new_chunk: bool
    inference_ms: float


class PolicyRunner(Protocol):
    """Small common interface used by the simulator evaluator."""

    expected_input_keys: tuple[str, ...]
    output_action_dim: int | None

    def reset(self) -> None: ...

    def select_action(self, observation: Mapping[str, Any]) -> InferenceSample: ...

    def close(self) -> None: ...


def _as_batched_action(value: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Extract ``action`` and normalize it to ``(B, A)`` or ``(B, T, A)``."""
    if isinstance(value, Mapping):
        if ACTION_KEY not in value:
            raise KeyError("Policy output mapping does not contain the 'action' key.")
        value = value[ACTION_KEY]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim not in (2, 3):
        raise ValueError(
            "Policy action must have shape (A), (B, A), or (B, T, A); "
            f"received {tuple(value.shape)}."
        )
    if not torch.isfinite(value).all():
        raise ValueError("Policy action contains NaN or infinity.")
    if device is not None:
        value = value.to(device=device)
    return value.float()


def camera_rgb_to_lerobot(rgb: torch.Tensor) -> torch.Tensor:
    """Convert Isaac camera RGB/RGBA ``BHWC`` into LeRobot ``BCHW`` float RGB."""
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.as_tensor(rgb)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
        raise ValueError(
            "Camera RGB must have shape (H, W, 3/4) or (B, H, W, 3/4); "
            f"received {tuple(rgb.shape)}."
        )
    rgb = rgb[..., :3]
    if rgb.dtype == torch.uint8:
        rgb = rgb.float().div_(255.0)
    else:
        rgb = rgb.float()
        # Isaac camera outputs are normally uint8.  This also supports float
        # render products that still use the 0..255 range.
        if rgb.numel() and float(rgb.detach().amax().item()) > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
    return rgb.permute(0, 3, 1, 2).contiguous()


def adapt_policy_action(
    policy_action: Any,
    *,
    current_joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scales: torch.Tensor,
    mode: str = "manager",
    clip: float | None = 1.0,
) -> ActionAdaptation:
    """Convert decoded LeRobot actions into the 15-D Isaac manager command.

    ``manager`` passes through the action representation stored in the
    recommended dataset. ``joint_position`` accepts absolute radians, while
    ``joint_delta`` accepts a delta in radians relative to the current state.
    """
    if mode not in ACTION_MODES:
        raise ValueError(f"Unknown action mode {mode!r}; choose one of {ACTION_MODES}.")
    action = _as_batched_action(policy_action, device=current_joint_pos.device)
    if action.ndim != 2:
        raise ValueError("adapt_policy_action expects one action step with shape (B, A).")
    expected_dim = len(CONTROLLED_JOINT_NAMES)
    if action.shape[-1] != expected_dim:
        raise ValueError(
            f"GR00T action dimension is {action.shape[-1]}, but the Kuavo manager requires "
            f"{expected_dim}: {', '.join(CONTROLLED_JOINT_NAMES)}."
        )
    for label, tensor in (
        ("current_joint_pos", current_joint_pos),
        ("default_joint_pos", default_joint_pos),
        ("action_scales", action_scales),
    ):
        if tensor.shape != action.shape:
            raise ValueError(
                f"{label} shape {tuple(tensor.shape)} does not match action shape {tuple(action.shape)}."
            )
    if torch.any(action_scales == 0):
        raise ValueError("Action scales must be non-zero.")

    if mode == "manager":
        manager_action = action
    elif mode == "joint_position":
        manager_action = (action - default_joint_pos) / action_scales
    else:
        manager_action = (current_joint_pos + action - default_joint_pos) / action_scales

    unclipped = manager_action.clone()
    saturation_fraction = 0.0
    if clip is not None:
        if clip <= 0.0:
            raise ValueError("Action clip must be positive or None.")
        saturated = torch.abs(manager_action) > clip
        saturation_fraction = float(saturated.float().mean().item())
        manager_action = manager_action.clamp(-clip, clip)
    return ActionAdaptation(manager_action, unclipped, saturation_fraction)


def adapt_manager_action(
    policy_action: Any,
    *,
    expected_dim: int,
    device: torch.device | str,
    clip: float | None = 1.0,
) -> ActionAdaptation:
    """Validate and clip an action already expressed in manager coordinates."""
    action = _as_batched_action(policy_action, device=device)
    if action.ndim != 2:
        raise ValueError("Manager action expects one action step with shape (B, A).")
    if action.shape[-1] != expected_dim:
        raise ValueError(
            f"GR00T action dimension is {action.shape[-1]}, but this configured manager requires "
            f"{expected_dim}. Use the same gripper preset used while collecting/training."
        )
    unclipped = action.clone()
    saturation_fraction = 0.0
    if clip is not None:
        if clip <= 0.0:
            raise ValueError("Action clip must be positive or None.")
        saturated = torch.abs(action) > clip
        saturation_fraction = float(saturated.float().mean().item())
        action = action.clamp(-clip, clip)
    return ActionAdaptation(action, unclipped, saturation_fraction)


def adapt_rwh_kuavo_v2_action(
    policy_action: Any,
    *,
    current_joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scales: torch.Tensor,
    joint_limits: torch.Tensor | None = None,
    clip: float | None = 1.0,
) -> ActionAdaptation:
    """Map the RwH 16-D raw-joint schema onto the Kuavo 17-D manager action.

    The policy order is left arm 7, left claw, right arm 7, right claw.  The
    manager order is waist, left arm 7, right arm 7, left gripper, right
    gripper.  Waist is held at its current position and claw values use the
    dataset convention 0=open, 1=closed.
    """
    action = _as_batched_action(policy_action, device=current_joint_pos.device)
    if action.ndim != 2 or action.shape[-1] != len(RWH_KUAVO_V2_NAMES):
        raise ValueError(
            "RwH-Kuavo V2 action must have shape (B, 16) in left7/claw/right7/claw order; "
            f"received {tuple(action.shape)}."
        )
    for label, tensor in (
        ("current_joint_pos", current_joint_pos),
        ("default_joint_pos", default_joint_pos),
        ("action_scales", action_scales),
    ):
        expected = (action.shape[0], len(CONTROLLED_JOINT_NAMES))
        if tensor.shape != expected:
            raise ValueError(f"{label} shape {tuple(tensor.shape)} does not match {expected}.")
    if torch.any(action_scales == 0):
        raise ValueError("Action scales must be non-zero.")

    left_target = action[:, :7]
    right_target = action[:, 8:15]
    limited = torch.zeros(
        (action.shape[0], len(CONTROLLED_JOINT_NAMES) + 2),
        dtype=torch.bool,
        device=action.device,
    )
    if joint_limits is not None:
        expected_limits = (
            action.shape[0],
            len(CONTROLLED_JOINT_NAMES),
            2,
        )
        if joint_limits.shape != expected_limits:
            raise ValueError(
                f"joint_limits shape {tuple(joint_limits.shape)} does not match {expected_limits}."
            )
        left_limits = joint_limits[:, 1:8]
        right_limits = joint_limits[:, 8:15]
        limited[:, 1:8] = (left_target < left_limits[..., 0]) | (
            left_target > left_limits[..., 1]
        )
        limited[:, 8:15] = (right_target < right_limits[..., 0]) | (
            right_target > right_limits[..., 1]
        )
        left_target = torch.clamp(
            left_target, left_limits[..., 0], left_limits[..., 1]
        )
        right_target = torch.clamp(
            right_target, right_limits[..., 0], right_limits[..., 1]
        )
    waist_hold = (current_joint_pos[:, :1] - default_joint_pos[:, :1]) / action_scales[:, :1]
    left_manager = (left_target - default_joint_pos[:, 1:8]) / action_scales[:, 1:8]
    right_manager = (right_target - default_joint_pos[:, 8:15]) / action_scales[:, 8:15]
    left_claw = action[:, 7:8].clamp(0.0, 1.0)
    right_claw = action[:, 15:16].clamp(0.0, 1.0)
    limited[:, 15:16] = action[:, 7:8] != left_claw
    limited[:, 16:17] = action[:, 15:16] != right_claw
    left_gripper = 1.0 - 2.0 * left_claw
    right_gripper = 1.0 - 2.0 * right_claw
    manager_action = torch.cat(
        (waist_hold, left_manager, right_manager, left_gripper, right_gripper), dim=-1
    )
    unclipped = manager_action.clone()
    if clip is not None:
        if clip <= 0.0:
            raise ValueError("Action clip must be positive or None.")
        limited |= torch.abs(manager_action) > clip
        manager_action = manager_action.clamp(-clip, clip)
    saturation_fraction = float(limited.float().mean().item())
    return ActionAdaptation(manager_action, unclipped, saturation_fraction)


class KuavoLeRobotBridge:
    """Build LeRobot observations and apply its decoded action convention."""

    @property
    def end_effector_pose_w(self):
        """[env, left/right, xyz+wxyz] calibrated closed TCP; not added to policy input."""
        from ..robots.end_effector import get_end_effector_frames
        return get_end_effector_frames(self.robot).center_pose_w

    def __init__(
        self,
        env: Any,
        *,
        camera_map: Mapping[str, str] | None = None,
        state_mode: str = "manager",
        action_mode: str = "manager",
        action_clip: float | None = 1.0,
        policy_profile: str = DEFAULT_POLICY_PROFILE,
        gripper_settings: Any | None = None,
    ) -> None:
        if state_mode not in STATE_MODES:
            raise ValueError(f"Unknown state mode {state_mode!r}; choose one of {STATE_MODES}.")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"Unknown action mode {action_mode!r}; choose one of {ACTION_MODES}.")
        if policy_profile not in POLICY_PROFILES:
            raise ValueError(
                f"Unknown policy profile {policy_profile!r}; choose one of {POLICY_PROFILES}."
            )
        if policy_profile in ARM_CLAW_PROFILES:
            if state_mode != "joint_position" or action_mode != "joint_position":
                raise ValueError(
                    "Arm/claw profiles require raw joint_position state and actions."
                )
        self.env = env
        self.robot = env.scene["robot"]
        # Joint-action policy schemas remain unchanged. Cartesian consumers can
        # explicitly query the calibrated TCP through end_effector_pose_w below.
        self.policy_profile = policy_profile
        action_manager = getattr(env, "action_manager", None)
        self.manager_action_dim = int(
            getattr(action_manager, "total_action_dim", len(CONTROLLED_JOINT_NAMES))
        )
        try:
            joint_ids, joint_names = self.robot.find_joints(
                list(CONTROLLED_JOINT_NAMES), preserve_order=True
            )
        except TypeError:
            joint_ids, joint_names = self.robot.find_joints(list(CONTROLLED_JOINT_NAMES))
        if tuple(joint_names) != CONTROLLED_JOINT_NAMES:
            raise RuntimeError(
                "Kuavo controlled joint order differs from the LeRobot schema. "
                f"Expected {CONTROLLED_JOINT_NAMES}, resolved {tuple(joint_names)}."
            )
        self.joint_ids = list(joint_ids)
        default_camera_map = (
            RWH_KUAVO_V2_CAMERA_MAP if policy_profile in ARM_CLAW_PROFILES else DEFAULT_CAMERA_MAP
        )
        self.camera_map = dict(default_camera_map if camera_map is None else camera_map)
        if not self.camera_map:
            raise ValueError("At least one camera must be mapped for GR00T evaluation.")
        for policy_key, scene_key in self.camera_map.items():
            if not policy_key.startswith("observation.images."):
                raise ValueError(
                    f"Camera policy key {policy_key!r} must start with 'observation.images.'."
                )
            try:
                env.scene[scene_key]
            except KeyError as exc:
                raise KeyError(
                    f"Isaac scene has no camera {scene_key!r} requested for {policy_key!r}."
                ) from exc
        self.state_mode = state_mode
        self.action_mode = action_mode
        self.action_clip = action_clip
        self._scales_1d = torch.tensor(
            MANAGER_ACTION_SCALES,
            device=self.robot.device,
            dtype=self.robot.data.joint_pos.dtype,
        )
        self.grippers = resolve_gripper_views(env, gripper_settings)
        self.manager_action_names = (
            *CONTROLLED_JOINT_NAMES, *(f"{side}_gripper" for side in self.grippers)
        )
        if len(self.manager_action_names) != self.manager_action_dim:
            raise RuntimeError(
                f"Manager action dimension {self.manager_action_dim} does not match "
                f"upper body plus configured grippers: {self.manager_action_names}."
            )
        if self.policy_profile in ARM_CLAW_PROFILES:
            if self.manager_action_dim != 17:
                raise RuntimeError(
                    "Arm/claw profiles require the 17-D Kuavo manager "
                    f"(waist + 14 arm + 2 claw), received {self.manager_action_dim}."
                )
            if gripper_settings is None or tuple(self.grippers) != ("left", "right"):
                raise RuntimeError(
                    "Arm/claw profiles require two configured grippers with open/close commands."
                )

    @property
    def action_dim(self) -> int:
        if self.policy_profile in ARM_CLAW_PROFILES:
            return len(RWH_KUAVO_V2_NAMES)
        return self.manager_action_dim if self.action_mode == "manager" else len(self.joint_ids)

    @property
    def state_names(self) -> tuple[str, ...]:
        if self.policy_profile in ARM_CLAW_PROFILES:
            return RWH_KUAVO_V2_NAMES
        names = list(CONTROLLED_JOINT_NAMES)
        for gripper in self.grippers.values():
            names.extend(gripper.state_names)
        return tuple(names)

    @property
    def action_names(self) -> tuple[str, ...]:
        if self.policy_profile in ARM_CLAW_PROFILES:
            return RWH_KUAVO_V2_NAMES
        return self.manager_action_names if self.action_mode == "manager" else CONTROLLED_JOINT_NAMES

    def _joint_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.robot.data.joint_pos[:, self.joint_ids]
        defaults = self.robot.data.default_joint_pos[:, self.joint_ids]
        scales = self._scales_1d.unsqueeze(0).expand_as(current)
        return current, defaults, scales

    def state(self) -> torch.Tensor:
        """Return the policy state without copying any camera observations."""
        current, defaults, scales = self._joint_tensors()
        if self.policy_profile in ARM_CLAW_PROFILES:
            left_claw = self.grippers["left"].claw_state()
            right_claw = self.grippers["right"].claw_state()
            return torch.cat(
                (current[:, 1:8], left_claw, current[:, 8:15], right_claw), dim=-1
            )
        elif self.state_mode == "manager":
            state = (current - defaults) / scales
        else:
            state = current
        gripper_states = []
        for gripper in self.grippers.values():
            gripper_states.append(gripper.joint_state(relative=self.state_mode == "manager"))
        if gripper_states:
            state = torch.cat((state, *gripper_states), dim=-1)
        return state.clone()

    def observation(self, task: str) -> dict[str, Any]:
        state = self.state()
        observation: dict[str, Any] = {STATE_KEY: state.clone(), "task": [task] * state.shape[0]}
        for policy_key, scene_key in self.camera_map.items():
            camera = self.env.scene[scene_key]
            observation[policy_key] = camera_rgb_to_lerobot(camera.data.output["rgb"])
        return observation

    def action(self, policy_action: Any) -> ActionAdaptation:
        if self.policy_profile in ARM_CLAW_PROFILES:
            current, defaults, scales = self._joint_tensors()
            return adapt_rwh_kuavo_v2_action(
                policy_action,
                current_joint_pos=current,
                default_joint_pos=defaults,
                action_scales=scales,
                joint_limits=self.robot.data.soft_joint_pos_limits[:, self.joint_ids],
                clip=self.action_clip,
            )
        if self.action_mode == "manager":
            return adapt_manager_action(
                policy_action,
                expected_dim=self.manager_action_dim,
                device=self.robot.device,
                clip=self.action_clip,
            )
        current, defaults, scales = self._joint_tensors()
        converted = adapt_policy_action(
            policy_action,
            current_joint_pos=current,
            default_joint_pos=defaults,
            action_scales=scales,
            mode=self.action_mode,
            clip=self.action_clip,
        )
        extra_dim = self.manager_action_dim - len(self.joint_ids)
        if extra_dim < 0:
            raise RuntimeError(
                f"Manager exposes {self.manager_action_dim} actions, fewer than the required "
                f"{len(self.joint_ids)} Kuavo joints."
            )
        if extra_dim == 0:
            return converted
        # Absolute/delta legacy policies do not predict binary grippers. Hold
        # every added gripper open (positive convention) in that mode.
        extra = torch.ones(
            (converted.action.shape[0], extra_dim),
            device=converted.action.device,
            dtype=converted.action.dtype,
        )
        return ActionAdaptation(
            torch.cat((converted.action, extra), dim=-1),
            torch.cat((converted.unclipped_action, extra), dim=-1),
            converted.saturation_fraction,
        )

    def hold_action(self) -> ActionAdaptation:
        """Build a manager command that holds the robot's current arm state.

        This is intended for a short post-reset stabilization interval before
        policy rollout. Arm/claw profiles retain their measured claw
        fraction; legacy manager configurations keep any extra grippers open.
        """
        current, defaults, scales = self._joint_tensors()
        if self.policy_profile in ARM_CLAW_PROFILES:
            return self.action(self.state())
        if self.action_mode == "joint_position":
            return self.action(current)
        if self.action_mode == "joint_delta":
            return self.action(torch.zeros_like(current))

        manager_action = (current - defaults) / scales
        extra_dim = self.manager_action_dim - manager_action.shape[1]
        if extra_dim < 0:
            raise RuntimeError(
                f"Manager exposes {self.manager_action_dim} actions, fewer than the required "
                f"{manager_action.shape[1]} Kuavo joints."
            )
        if extra_dim:
            manager_action = torch.cat(
                (
                    manager_action,
                    torch.ones(
                        (manager_action.shape[0], extra_dim),
                        device=manager_action.device,
                        dtype=manager_action.dtype,
                    ),
                ),
                dim=-1,
            )
        return ActionAdaptation(manager_action, manager_action.clone(), 0.0)


class LeRobotGrootRunner:
    """Load a LeRobot GR00T checkpoint and execute decoded action chunks."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        actions_per_inference: int | None = None,
        expected_action_dim: int | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        configured_steps = int(getattr(policy.config, "n_action_steps", 1))
        self.actions_per_inference = actions_per_inference or configured_steps
        if self.actions_per_inference <= 0:
            raise ValueError("actions_per_inference must be positive.")
        input_features = getattr(policy.config, "input_features", {}) or {}
        self.expected_input_keys = tuple(input_features)
        output_features = getattr(policy.config, "output_features", {}) or {}
        action_feature = output_features.get(ACTION_KEY)
        self.output_action_dim = (
            int(action_feature.shape[0]) if action_feature is not None else None
        )
        self.expected_action_dim = expected_action_dim
        self._queue: deque[torch.Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: str,
        actions_per_inference: int | None = None,
        local_files_only: bool = False,
        expected_action_dim: int | None = None,
        base_model_path: str | None = None,
        strict: bool = True,
    ) -> "LeRobotGrootRunner":
        try:
            from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: F401
            try:
                from lerobot.configs import PreTrainedConfig
            except ImportError:
                from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.groot.modeling_groot import GrootPolicy
        except (ImportError, ModuleNotFoundError) as exc:
            version = "unknown"
            try:
                import lerobot

                version = getattr(lerobot, "__version__", "unknown")
            except ImportError:
                pass
            raise RuntimeError(
                "This evaluator requires a LeRobot build that supports this checkpoint's "
                f"GR00T generation. The imported LeRobot version is {version!r}; verify "
                "that `lerobot.policies.groot` imports successfully. GR00T N1.5 requires "
                "LeRobot 0.5.x, while N1.7 uses a newer compatible checkout."
            ) from exc

        config = PreTrainedConfig.from_pretrained(
            checkpoint,
            local_files_only=local_files_only,
        )
        if getattr(config, "type", None) != "groot":
            raise ValueError(
                f"Checkpoint policy type is {getattr(config, 'type', None)!r}; expected 'groot'."
            )
        config.device = device
        configured_base = str(getattr(config, "base_model_path", ""))
        if base_model_path:
            config.base_model_path = base_model_path
        elif (
            configured_base
            and not Path(configured_base).exists()
            and "GR00T-N1.5-3B" in configured_base
        ):
            config.base_model_path = "nvidia/GR00T-N1.5-3B"
        policy = GrootPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=local_files_only,
            strict=strict,
        )
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            actions_per_inference=actions_per_inference,
            expected_action_dim=expected_action_dim,
        )

    def reset(self) -> None:
        self._queue.clear()
        self.policy.reset()

    def close(self) -> None:
        return None

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        missing = tuple(key for key in self.expected_input_keys if key not in observation)
        if missing:
            raise KeyError(
                "The checkpoint expects observation keys that the Kuavo bridge did not provide: "
                f"{missing}. Match training camera names with repeated "
                "`--camera-map POLICY_KEY=SCENE_CAMERA` arguments."
            )
        if self.expected_action_dim is not None and self.output_action_dim not in (
            None,
            self.expected_action_dim,
        ):
            raise ValueError(
                f"Checkpoint action dimension is {self.output_action_dim}; this environment requires "
                f"{self.expected_action_dim}. Use the same gripper preset used for training."
            )

    def select_action(self, observation: Mapping[str, Any]) -> InferenceSample:
        if self._queue:
            return InferenceSample(self._queue.popleft(), False, 0.0)
        self.validate_observation(observation)
        start = time.perf_counter()
        with torch.inference_mode():
            batch = self.preprocessor(dict(observation))
            raw_chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(raw_chunk)
            raw_decoded = _as_batched_action(raw_chunk)
            decoded = _as_batched_action(chunk)
            # LeRobot 0.5.x's GR00T N1.5 postprocessor selects the last
            # timestep when handed a full (B,T,A) chunk.  Preserve the chunk
            # by postprocessing each timestep independently in that case.
            if raw_decoded.ndim == 3 and decoded.ndim == 2 and raw_decoded.shape[1] > 1:
                processed_steps = [
                    _as_batched_action(self.postprocessor(raw_decoded[:, index]))
                    for index in range(raw_decoded.shape[1])
                ]
                if not all(step.ndim == 2 for step in processed_steps):
                    raise RuntimeError(
                        "GR00T per-step postprocessor returned an invalid action shape."
                    )
                decoded = torch.stack(processed_steps, dim=1)
        policy_device = str(getattr(self.policy.config, "device", ""))
        if torch.cuda.is_available() and policy_device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(policy_device))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if decoded.ndim == 2:
            decoded = decoded.unsqueeze(1)
        count = min(self.actions_per_inference, decoded.shape[1])
        if count <= 0:
            raise RuntimeError("GR00T returned an empty action chunk.")
        self._queue.extend(decoded[:, index].detach() for index in range(count))
        return InferenceSample(self._queue.popleft(), True, elapsed_ms)


_IPC_HEADER = struct.Struct("!Q")


def send_framed_pickle(connection: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(_IPC_HEADER.pack(len(data)))
    connection.sendall(data)


def receive_framed_pickle(connection: socket.socket) -> Any:
    def receive_exact(size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise EOFError("GR00T policy worker closed its IPC socket unexpectedly.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    (size,) = _IPC_HEADER.unpack(receive_exact(_IPC_HEADER.size))
    return pickle.loads(receive_exact(size))


class SubprocessLeRobotGrootRunner:
    """Run a GR00T policy in a separate LeRobot Conda Python process."""

    def __init__(
        self,
        *,
        python_executable: str,
        checkpoint: str,
        device: str,
        actions_per_inference: int | None = None,
        local_files_only: bool = False,
        expected_action_dim: int | None = None,
        base_model_path: str | None = None,
        strict: bool = True,
    ) -> None:
        python_path = Path(python_executable).expanduser().resolve()
        if not python_path.is_file():
            raise FileNotFoundError(f"LeRobot Python executable does not exist: {python_path}")
        parent_socket, child_socket = socket.socketpair()
        command = [
            str(python_path),
            "-m",
            "kuavo_isaaclab_scene.evaluation.groot_policy_worker",
            "--ipc-fd",
            str(child_socket.fileno()),
            "--checkpoint",
            checkpoint,
            "--device",
            device,
        ]
        if actions_per_inference is not None:
            command.extend(("--actions-per-inference", str(actions_per_inference)))
        if local_files_only:
            command.append("--local-files-only")
        if expected_action_dim is not None:
            command.extend(("--expected-action-dim", str(expected_action_dim)))
        if base_model_path:
            command.extend(("--base-model-path", base_model_path))
        if not strict:
            command.append("--no-strict")
        child_env = os.environ.copy()
        package_path = str(PACKAGE_IMPORT_ROOT)
        existing_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (
            f"{package_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else package_path
        )
        self._socket = parent_socket
        self._process = subprocess.Popen(
            command,
            env=child_env,
            pass_fds=(child_socket.fileno(),),
        )
        child_socket.close()
        try:
            ready = receive_framed_pickle(self._socket)
        except Exception:
            self.close()
            raise
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(ready.get("error", "GR00T policy worker failed during startup."))
        self.expected_input_keys = tuple(ready["expected_input_keys"])
        self.input_shapes = {
            str(key): tuple(int(size) for size in shape)
            for key, shape in ready.get("input_shapes", {}).items()
        }
        self.output_action_dim = ready.get("output_action_dim")
        self.lerobot_version = str(ready.get("lerobot_version", "unknown"))

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        send_framed_pickle(self._socket, dict(payload))
        response = receive_framed_pickle(self._socket)
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "GR00T policy worker request failed."))
        return response

    def reset(self) -> None:
        self._request({"op": "reset"})

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        missing = tuple(key for key in self.expected_input_keys if key not in observation)
        if missing:
            raise KeyError(f"Checkpoint observation keys are missing: {missing}.")
        for key, expected_shape in self.input_shapes.items():
            if key not in observation or not hasattr(observation[key], "shape"):
                continue
            actual = tuple(int(size) for size in observation[key].shape)
            if actual and actual[0] == 1:
                actual = actual[1:]
            if actual != expected_shape:
                raise ValueError(
                    f"Checkpoint expects {key} shape {expected_shape}, but bridge produced {actual}."
                )

    def select_action(self, observation: Mapping[str, Any]) -> InferenceSample:
        self.validate_observation(observation)
        cpu_observation = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in observation.items()
        }
        response = self._request({"op": "select_action", "observation": cpu_observation})
        return InferenceSample(
            action=torch.as_tensor(response["action"]),
            inferred_new_chunk=bool(response["inferred_new_chunk"]),
            inference_ms=float(response["inference_ms"]),
        )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        connection = getattr(self, "_socket", None)
        if process is None:
            return
        if process.poll() is None and connection is not None:
            try:
                send_framed_pickle(connection, {"op": "close"})
                receive_framed_pickle(connection)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if connection is not None:
            connection.close()
            self._socket = None
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._process = None


class ZeroLeRobotPolicyRunner:
    """Dependency-free policy used to smoke-test the complete simulator loop."""

    expected_input_keys = (STATE_KEY, *DEFAULT_CAMERA_MAP)

    def __init__(self, *, action_dim: int = len(CONTROLLED_JOINT_NAMES)) -> None:
        self.action_dim = action_dim
        self.output_action_dim = action_dim

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def select_action(self, observation: Mapping[str, Any]) -> InferenceSample:
        state = observation[STATE_KEY]
        action = torch.zeros((state.shape[0], self.action_dim), device=state.device)
        return InferenceSample(action, True, 0.0)


def parse_camera_map(
    values: Sequence[str] | None, *, default_map: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Parse repeated ``POLICY_KEY=SCENE_CAMERA`` CLI values."""
    if not values:
        return dict(default_map or DEFAULT_CAMERA_MAP)
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid camera mapping {value!r}; expected POLICY_KEY=SCENE_CAMERA."
            )
        policy_key, scene_key = (part.strip() for part in value.split("=", 1))
        if not policy_key or not scene_key:
            raise ValueError(
                f"Invalid camera mapping {value!r}; both sides must be non-empty."
            )
        if not policy_key.startswith("observation.images."):
            policy_key = f"observation.images.{policy_key}"
        if policy_key in result:
            raise ValueError(f"Camera policy key {policy_key!r} is mapped more than once.")
        result[policy_key] = scene_key
    return result
