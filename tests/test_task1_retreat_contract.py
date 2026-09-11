import numpy as np

from data_collection.task1.contract import WAIST_ARM_JOINT_NAMES
from data_collection.task1.retreat import waist_preserving_retreat


def test_retreat_holds_selected_waist_for_every_arm_waypoint():
    arms = np.arange(42, dtype=float).reshape(3, 14)

    names, rows = waist_preserving_retreat([0.12, -0.03], arms)

    assert names == WAIST_ARM_JOINT_NAMES
    np.testing.assert_allclose(rows[:, :2], [[0.12, -0.03]] * 3)
    np.testing.assert_allclose(rows[:, 2:], arms)
