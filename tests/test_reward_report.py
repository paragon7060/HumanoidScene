"""Pure formatting checks: no Isaac Sim, XR, or training process."""

import unittest
from kuavo_isaaclab_scene.rl.debug.reward_report import step_contributions, format_report, reward_summary


class RewardReportTests(unittest.TestCase):
    def test_weighted_rate_and_discrete_bonus(self):
        terms = step_contributions([("lift", [5.]), ("success", [4500.]), ("collision", [-2.])], 1 / 30)
        self.assertAlmostEqual(terms["lift"], 1 / 6)
        self.assertAlmostEqual(terms["success"], 150.)
        self.assertAlmostEqual(terms["collision"], -2 / 30)

    def test_zero_term_is_visible(self):
        self.assertEqual(step_contributions([("lift", [0.])], 1 / 30), {"lift": 0.})

    def test_terminal_is_not_replaced_with_reset_values(self):
        sample = dict(terms={"success": 150.}, total=150., lift_cm=6.2, hold=.5,
                      left_grasp=False, right_grasp=True, left_distance_cm=10., right_distance_cm=1.)
        output = format_report(sample, "SUCCESS - last step", 154.)
        self.assertIn("success: +150.00000", output)
        self.assertIn("6.2 cm", output)
        self.assertIn("Grasp L/R: 0/1", output)
        self.assertIn("RETURN: +154.000", output)
        sample.update(blocked_checks=[], obstacle_force=500., collision_constraints_enabled=False)
        output = format_report(sample, "SUCCESS", 154.)
        self.assertIn("500.000 N", output)
        self.assertIn("Collision termination/penalty: OFF", output)

    def test_no_physics_yet(self):
        self.assertIn("Waiting for first physics step", format_report(None, "PAUSED", 0.))

    def test_large_terminal_summary_preserves_reason_and_unmet_hold(self):
        sample = dict(failure=True, failure_reasons=["obstacle_collision"], obstacle_force=21.5,
                      obstacle_limit=20., lift_cm=7., hold=.2, tilt_deg=10.,
                      success_checks=dict(grasp=True, height=True, tilt=True, hold=False))
        headline, checks = reward_summary(sample, "FAILURE")
        self.assertEqual(headline, "FAILED: OBSTACLE COLLISION")
        self.assertTrue(any("21.50 N > 20 N" in line for line in checks))
        self.assertIn("NEED: Hold 0.20 / 0.5 s", checks)
        self.assertFalse(any("speed" in line or "contact" in line for line in checks))
        sample.update(failure=False, failure_reasons=[], success=True)
        self.assertEqual(reward_summary(sample, "SUCCESS")[0], "SUCCESS")

    def test_grasp_diagnostics_and_unmet_success_conditions(self):
        sample = dict(terms={}, total=0., lift_cm=7., hold=0.,
                      left_grasp=False, right_grasp=True, left_distance_cm=10., right_distance_cm=1.,
                      blocked_checks=["speed"], grasp_debug=["R flap_left: raw=0 held=1"])
        output = format_report(sample, "RUN", 0.)
        self.assertIn("CHECK: speed", output)
        self.assertIn("raw=0 held=1", output)
        sample["blocked_checks"] = []
        self.assertIn("CHECK: pass - keep holding", format_report(sample, "RUN", 0.))
