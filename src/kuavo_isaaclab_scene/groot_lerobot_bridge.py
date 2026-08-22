"""LeRobot/GR00T N1.7 bridge for the Kuavo ManagerBased environment.

This module keeps policy I/O separate from the Isaac Lab application launcher so
its action conversion and image formatting can be unit-tested without starting
Isaac Sim.  The public LeRobot keys intentionally match the dataset convention:
``observation.state``, ``observation.images.<camera>``, ``task``, and ``action``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Mapping, Protocol, Sequence

import torch


CONTROLLED_JOINT_NAMES = (
    "waist_yaw_joint",
    *(f"zarm_l{index}_joint" for index in range(1, 8)),
    *(f"zarm_r{index}_joint" for index in range(1, 8)),
)
MANAGER_ACTION_SCALES = (1.0, *(0.45 for _ in range(14)))
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_CAMERA_MAP = {
    "observation.images.head": "robustness_camera",
    "observation.images.waist": "waist_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
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


class KuavoLeRobotBridge:
    """Build LeRobot observations and apply its decoded action convention."""

    def __init__(
        self,
        env: Any,
        *,
        camera_map: Mapping[str, str] | None = None,
        state_mode: str = "manager",
        action_mode: str = "manager",
        action_clip: float | None = 1.0,
    ) -> None:
        if state_mode not in STATE_MODES:
            raise ValueError(f"Unknown state mode {state_mode!r}; choose one of {STATE_MODES}.")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"Unknown action mode {action_mode!r}; choose one of {ACTION_MODES}.")
        self.env = env
        self.robot = env.scene["robot"]
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
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
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
        self.gripper_state_sources: list[tuple[str, Any, list[int], tuple[str, ...]]] = []
        for side in ("left", "right"):
            try:
                gripper = env.scene[f"{side}_gripper"]
            except KeyError:
                continue
            try:
                ids, names = gripper.find_joints([".*"], preserve_order=True)
            except TypeError:
                ids, names = gripper.find_joints([".*"])
            self.gripper_state_sources.append((side, gripper, list(ids), tuple(names)))

    @property
    def action_dim(self) -> int:
        return self.manager_action_dim if self.action_mode == "manager" else len(self.joint_ids)

    @property
    def state_names(self) -> tuple[str, ...]:
        names = list(CONTROLLED_JOINT_NAMES)
        for side, _, _, joint_names in self.gripper_state_sources:
            names.extend(f"{side}_{name}" for name in joint_names)
        return tuple(names)

    def _joint_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.robot.data.joint_pos[:, self.joint_ids]
        defaults = self.robot.data.default_joint_pos[:, self.joint_ids]
        scales = self._scales_1d.unsqueeze(0).expand_as(current)
        return current, defaults, scales

    def observation(self, task: str) -> dict[str, Any]:
        current, defaults, scales = self._joint_tensors()
        if self.state_mode == "manager":
            state = (current - defaults) / scales
        else:
            state = current
        gripper_states = []
        for _, gripper, joint_ids, _ in self.gripper_state_sources:
            values = gripper.data.joint_pos[:, joint_ids]
            if self.state_mode == "manager":
                values = values - gripper.data.default_joint_pos[:, joint_ids]
            gripper_states.append(values)
        if gripper_states:
            state = torch.cat((state, *gripper_states), dim=-1)
        observation: dict[str, Any] = {STATE_KEY: state.clone(), "task": [task] * state.shape[0]}
        for policy_key, scene_key in self.camera_map.items():
            camera = self.env.scene[scene_key]
            observation[policy_key] = camera_rgb_to_lerobot(camera.data.output["rgb"])
        return observation

    def action(self, policy_action: Any) -> ActionAdaptation:
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


class LeRobotGrootRunner:
    """Load a LeRobot GR00T N1.7 checkpoint and execute decoded chunks."""

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
    ) -> "LeRobotGrootRunner":
        try:
            from lerobot.configs import PreTrainedConfig
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
                "This evaluator requires a current LeRobot build with GR00T N1.7 support. "
                f"The imported LeRobot version is {version!r}. Install/update with "
                "`pip install \"lerobot[groot]\"`, then verify that "
                "`lerobot.policies.groot` imports successfully."
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
        policy = GrootPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=local_files_only,
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
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)
        policy_device = str(getattr(self.policy.config, "device", ""))
        if torch.cuda.is_available() and policy_device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(policy_device))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        decoded = _as_batched_action(chunk)
        if decoded.ndim == 2:
            decoded = decoded.unsqueeze(1)
        count = min(self.actions_per_inference, decoded.shape[1])
        if count <= 0:
            raise RuntimeError("GR00T returned an empty action chunk.")
        self._queue.extend(decoded[:, index].detach() for index in range(count))
        return InferenceSample(self._queue.popleft(), True, elapsed_ms)


class ZeroLeRobotPolicyRunner:
    """Dependency-free policy used to smoke-test the complete simulator loop."""

    expected_input_keys = (STATE_KEY, *DEFAULT_CAMERA_MAP)
    def __init__(self, *, action_dim: int = len(CONTROLLED_JOINT_NAMES)) -> None:
        self.action_dim = action_dim
        self.output_action_dim = action_dim

    def reset(self) -> None:
        return None

    def select_action(self, observation: Mapping[str, Any]) -> InferenceSample:
        state = observation[STATE_KEY]
        action = torch.zeros((state.shape[0], self.action_dim), device=state.device)
        return InferenceSample(action, True, 0.0)


def parse_camera_map(values: Sequence[str] | None) -> dict[str, str]:
    """Parse repeated ``POLICY_KEY=SCENE_CAMERA`` CLI values."""
    if not values:
        return dict(DEFAULT_CAMERA_MAP)
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
