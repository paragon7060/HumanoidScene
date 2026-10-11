"""Bounded rack-contact diagnostics from existing pre-reset scalar packets."""
from copy import deepcopy
import math


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class RackContactSummary:
    """Keep three packets per environment; do not query or advance physics."""

    def __init__(self, count, rack_failure_threshold_n):
        failure_force = _finite(rack_failure_threshold_n)
        if not isinstance(count, int) or count < 1 or failure_force is None or failure_force <= 0:
            raise ValueError('Positive environment count and rack threshold required')
        self.contract = dict(
            name='existing_pre_reset_rack_contact_summary_v1',
            warning_force_n=.1 * failure_force,
            failure_force_n=failure_force,
            threshold_changes_success_safety_reward=False,
            first_event='first_processed_control_step_strictly_above_warning',
            not_first_PhysX_substep_contact=True,
            scope='active_non_numerical_frames_after_initial_settling_before_autoreset',
            force_source='existing_filtered_rack_force_max_by_robot_body',
            raw_observations_actions_or_replay_stored=False,
            new_sensors_physics_queries_or_steps=False)
        self.records = [dict(observed_control_steps=0, invalid_force_samples=0,
            first_warning=None, first_held_warning=None, largest_measured_force=None)
            for _ in range(count)]
        self.last_steps = [0] * count

    def observe(self, environment, result):
        row = self.records[environment]
        step = result['steps']
        if step <= self.last_steps[environment]:
            return  # Repeated packets cannot become another sample or event.
        self.last_steps[environment] = step
        row['observed_control_steps'] += 1
        force = _finite(result.get('rack_peak_force_n'))
        if force is None or force < 0:
            row['invalid_force_samples'] += 1
            return
        phase = result.get('staged_base', {}).get('phase', 'unrecorded')
        peak = row['largest_measured_force']
        keep_peak = peak is None or force > peak['max_rack_force_n']
        warning = force > self.contract['warning_force_n']
        keep_first = warning and row['first_warning'] is None
        keep_held = warning and phase == 'held_grasp' and row['first_held_warning'] is None
        if not (keep_peak or keep_first or keep_held):
            return
        distance = result.get('flap_distances')
        packet = dict(control_step=step, stage=phase, max_rack_force_n=force,
            max_rack_force_body=result.get('rack_peak_body'),
            flap_distances_m=None if distance is None else [_finite(v) for v in distance],
            pinching_hands=deepcopy(result.get('pinching')),
            stable_hands=deepcopy(result.get('stable_hands')),
            opposing_flaps=bool(result.get('opposing_flaps', False)),
            hold_time_s=_finite(result.get('hold_time_s')),
            proof_lift=bool(result.get('proof_lift', False)),
            rack_clearance_m=_finite(result.get('rack_clearance_m')),
            success=bool(result.get('success', False)), unsafe=bool(result.get('unsafe', False)))
        if keep_peak:
            row['largest_measured_force'] = packet
        if keep_first:
            row['first_warning'] = packet
        if keep_held:
            row['first_held_warning'] = packet

    def snapshot(self, environment):
        return deepcopy(dict(contract=self.contract, **self.records[environment]))
