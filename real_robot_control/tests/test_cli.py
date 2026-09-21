from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_rl_prepare_and_inspect(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "actions.npy"
            output = Path(folder) / "trajectory.npz"
            np.save(source, np.zeros((4, 16), dtype=np.float32))
            prepared = subprocess.run(
                [sys.executable, str(ROOT / "prepare_rl_actions.py"), str(source), str(output)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            inspected = subprocess.run(
                [sys.executable, str(ROOT / "inspect_trajectory.py"), str(output)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertIn('"samples": 4', inspected.stdout)

    def test_live_motion_requires_exact_confirmation(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_robot.py"), "missing.npz", "--enable-motion"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("S63_CLEAR_AND_ESTOP_READY", result.stderr)


if __name__ == "__main__":
    unittest.main()
