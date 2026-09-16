import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("measure_reversals", Path(__file__).parents[1] / "measure_reversals.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReversalTests(unittest.TestCase):
    def test_preview_never_moves_or_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module.subprocess, "run") as send:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(["--session", directory]), 0)
            send.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_send_requires_recording_session(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module.subprocess, "run") as send:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main(["--session", directory, "--send"])
            send.assert_not_called()

    def test_measurements_are_explicit_and_unknowns_remain_unknown(self):
        self.assertEqual(module.parse_widths("51 49"), [51,49])
        self.assertEqual(module.parse_widths("? 0"), [None,0])
        for value in ("50", "nan 30", "10 inf", "-1 10", "201 10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.parse_widths(value)

    def test_failed_command_stops_without_retry_or_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "session.json").write_text(json.dumps({"host":"lab@example", "workspace":"/ws", "ros_master":"http://master:11311"}))
            (session / "messages.jsonl").touch()
            with patch.object(module.subprocess, "run", return_value=module.subprocess.CompletedProcess([], 1)) as send:
                with patch("builtins.input", return_value=""), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(module.main(["--session", str(session), "--send"]), 1)
                self.assertEqual(send.call_count, 1)
            rows = [json.loads(line) for line in (session / "reversal_measurements.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["kind"], "command_failed_or_uncertain")
            self.assertFalse(any(row["kind"] == "width_measurement" for row in rows))

    def test_check75_keeps_approach_commands_and_measures_only_requested_points(self):
        replies = []
        expected_commands = []
        for _, commands in module.CHECK75_CASES:
            for percent in commands:
                replies.append("")
                expected_commands.append(str(percent))
                if percent == 75:
                    replies.extend(["26 27", ""])
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "session.json").write_text(json.dumps({"host":"lab@example", "workspace":"/ws", "ros_master":"http://master:11311"}))
            (session / "messages.jsonl").touch()
            with patch.object(module.subprocess, "run", return_value=module.subprocess.CompletedProcess([], 0)) as send:
                with patch.object(module.time, "sleep"), patch("builtins.input", side_effect=replies), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(["--session", str(session), "--send", "--protocol", "check75", "--measure-only-75"]), 0)
                actual_commands = [call.args[0][call.args[0].index("--position")+1] for call in send.call_args_list]
                self.assertEqual(actual_commands, expected_commands)
            rows = [json.loads(line) for line in (session / "reversal_measurements.jsonl").read_text().splitlines()]
            widths = [row for row in rows if row["kind"] == "width_measurement"]
            self.assertEqual(len(widths), 2)
            self.assertTrue(all(row["command_percent"] == 75 for row in widths))
            self.assertEqual(len([row for row in rows if row["kind"] == "width_not_requested"]), 9)
