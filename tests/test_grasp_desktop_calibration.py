"""CPU-only tests: desktop saving must not touch physics or import Isaac Sim."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/rl/debug"
          / "calibrate_grasp_desktop.py")
SPEC = importlib.util.spec_from_file_location("desktop_calibration_save", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DesktopCalibrationSaveTests(unittest.TestCase):
    def test_round_trip_four_points_and_failed_save_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            offsets = {name: [.01, -.02, .03] for name in
                       ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")}
            calibration = SimpleNamespace(path=Path(directory) / "nested" / "points.json",
                                          model_name="kuavo_s200062", offsets=lambda: offsets)
            MODULE.save_points(calibration)
            saved = calibration.path.read_text()
            payload = json.loads(saved)
            self.assertEqual(payload["offsets"], offsets)
            self.assertEqual(payload["frame"], "finger_link_local")
            self.assertEqual(payload["robot_model"], calibration.model_name)
            offsets["r_f_finger"][0] = float("nan")
            with self.assertRaises(ValueError):
                MODULE.save_points(calibration)
            self.assertEqual(calibration.path.read_text(), saved)
            self.assertEqual(list(calibration.path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
