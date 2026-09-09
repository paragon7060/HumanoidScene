"""Pure formatting checks: no Isaac Sim, XR, or training process."""

import unittest
from kuavo_isaaclab_scene.rl.debug.reward_report import step_contributions, format_report


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

    def test_no_physics_yet(self):
        self.assertIn("Waiting for first physics step", format_report(None, "PAUSED", 0.))

    def test_grasp_diagnostics_and_unmet_success_conditions(self):
        sample = dict(terms={}, total=0., lift_cm=7., hold=0.,
                      left_grasp=False, right_grasp=True, left_distance_cm=10., right_distance_cm=1.,
                      blocked_checks=["speed"], grasp_debug=["R flap_left: raw=0 held=1"])
        output = format_report(sample, "RUN", 0.)
        self.assertIn("CHECK: speed", output)
        self.assertIn("raw=0 held=1", output)
        sample["blocked_checks"] = []
        self.assertIn("CHECK: pass - keep holding", format_report(sample, "RUN", 0.))
