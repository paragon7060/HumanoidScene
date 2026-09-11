"""Capture the terminal step before Isaac Lab's automatic scene reset."""

import math

from isaaclab.managers import RecorderTerm
from .reward_report import step_contributions


class RewardProbe(RecorderTerm):
    def record_post_step(self):
        env = self._env
        t = env.command_manager.get_term("workcell")
        box = int(t.reward_box[0].item())
        env._quest_reward_sample = {
            "terms": step_contributions(env.reward_manager.get_active_iterable_terms(0), env.step_dt),
            "total": float(env.reward_buf[0].item()),
            "lift_cm": float((t.centers[0, box, 2] - env.scene.env_origins[0, 2]
                              - t.initial_z[0, box]).item()) * 100,
            "hold": float(t.dwell[0].item()),
            "required_hold": float(t.spec.hold_seconds),
            "required_lift_cm": float(t.spec.lift_height) * 100,
            "tilt_deg": math.degrees(math.acos(float(t.upright[0, box].clamp(-1, 1)))),
            "max_tilt_deg": math.degrees(t.spec.max_tilt),
            "obstacle_limit": float(t.spec.obstacle_contact_force),
            "collision_constraints_enabled": t.spec.collision_constraints_enabled,
            "reach_score": t.reach_progress.previous[0].tolist(),
            "reach_active": t.reach_progress.initialized[0].tolist(),
            "reach_progress": t.reach_progress.delta[0].tolist(),
            "orientation_progress": t.flap_progress.orientation_delta[0].tolist(),
            "lift_progress": float(t.flap_progress.lift_delta[0].item()),
            "grasp_bonus_paid": bool(t.flap_progress.grasp_seen[0].item()),
            "grasp_bonus_event": bool(t.flap_progress.grasp_bonus[0].item()),
            "required_hands": "/".join("L" if i == 0 else "R" for i in t.spec.grasp_hand_indices),
            "success_checks": {name: bool(values[0].item()) for name, values in t.pick_checks.items()},
            "left_grasp": bool(t.hand_grasp_flags[0, 0].item()),
            "right_grasp": bool(t.hand_grasp_flags[0, 1].item()),
            "left_distance_cm": float(t.hand_target_distance[0, 0].item()) * 100,
            "right_distance_cm": float(t.hand_target_distance[0, 1].item()) * 100,
            "success": bool(t.success[0].item()),
            "failure": bool(t.failure[0].item()),
            "timeout": bool(env.reset_time_outs[0].item()),
            "blocked_checks": [name for name, values in t.pick_checks.items() if not bool(values[0].item())],
            "failure_reasons": [name for name, values in t.failure_checks.items() if bool(values[0].item())],
            "obstacle_force": float(t.obstacle_forces[0].amax().item()),
            "grasp_debug": [
                f"{'L' if hand == 0 else 'R'} {flap}: d={100*t.grasp_candidate_distance[0, hand, candidate].item():.1f}cm "
                f"axisErr={math.degrees(math.acos(float(t.grasp_candidate_alignment[0, hand, candidate].clamp(0, 1)))):.1f}deg "
                f"raw={int(t.grasp_candidate_raw[0, hand, candidate])} "
                f"held={int(t.grasp_candidate_held[0, hand, candidate])}\n"
                f"region={''.join(str(int(v)) for v in t.grasp_candidate_region[0, hand, candidate])} "
                f"opp={int(t.grasp_candidate_opposed[0, hand, candidate])} "
                f"F={t.grasp_candidate_force[0, hand, candidate, 0].item():.2f}/"
                f"{t.grasp_candidate_force[0, hand, candidate, 1].item():.2f}N "
                f"gap={t.grasp_candidate_missing_s[0, hand, candidate].item():.3f}s\n"
                f"slip={1000*t.grasp_candidate_slip_m[0, hand, candidate].item():.1f}mm "
                f"open+={1000*t.grasp_candidate_opening_m[0, hand, candidate].item():.1f}mm"
                for hand in t.spec.grasp_hand_indices for candidate, flap in enumerate(t.spec.grasp_flaps)],
        }
        # No trajectory accumulation or HDF5 export: just one small snapshot.
        return None, None
