"""Verify operator gating and limits without connecting to or moving a robot."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "control_real_gripper", Path(__file__).parents[1] / "control_real_gripper.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class RealGripperControlTests(unittest.TestCase):
    def test_preview_never_connects_or_creates_log(self):
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "session" / "commands.jsonl"
            output = io.StringIO()
            with patch.object(control.subprocess, "run") as remote, contextlib.redirect_stdout(output):
                self.assertEqual(control.main(["--position", "25", "--log", str(log)]), 0)
                remote.assert_not_called()
            self.assertFalse(log.exists())
            self.assertIn('"send": false', output.getvalue())

    def test_invalid_motion_input_never_connects(self):
        import contextlib
        import io

        cases = [
            ["--send"], ["--position", "-1", "--send"], ["--position", "101", "--send"],
            ["--position", "nan", "--send"], ["--position", "0", "25", "50", "--send"],
            ["--position", "25", "--effort", "3", "--send"],
            ["--position", "25", "--velocity", "101", "--send"],
            ["--position", "25", "--check", "--send"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.object(control.subprocess, "run") as remote:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    control.main(arguments)
                self.assertEqual(error.exception.code, 2)
                remote.assert_not_called()

    def test_timeout_records_attempt_and_never_retries(self):
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "commands.jsonl"
            with patch.object(control.subprocess, "run", side_effect=control.subprocess.TimeoutExpired("ssh", 45)) as remote:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(control.main(["--position", "25", "--send", "--log", str(log)]), 1)
                self.assertEqual(remote.call_count, 1)
            self.assertIn('"kind": "timeout"', log.read_text())


if __name__ == "__main__":
    unittest.main()
