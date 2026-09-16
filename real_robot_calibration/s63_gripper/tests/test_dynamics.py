import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

FOLDER = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, FOLDER / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(FOLDER), *sys.path]):
        spec.loader.exec_module(module)
    return module


measure = load("measure_dynamics")
analyze = load("analyze_dynamics")


class DynamicsTests(unittest.TestCase):
    def session(self, path):
        (path / "session.json").write_text(json.dumps(dict(host="lab@example", workspace="/ws", ros_master="http://master:11311")))
        (path / "messages.jsonl").write_text(json.dumps(dict(kind="metadata")) + "\n" + json.dumps(dict(kind="message", topic="/leju_claw_state")) + "\n")

    def test_preview_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(measure.subprocess, "run") as send:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(["--session", directory, "--velocities", "25", "50", "75"]), 0)
            send.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failures_stop_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.session(path)
            with patch.object(measure.subprocess, "run", return_value=measure.subprocess.CompletedProcess([], 1)) as send, patch("builtins.input", return_value=""):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(measure.main(["--session", directory, "--send"]), 1)
            self.assertEqual(send.call_count, 1)
            rows = [json.loads(line) for line in (path / "dynamics_measurements.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["kind"], "command_failed_or_uncertain")

    def test_each_motion_is_gated_and_repeat_approach_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.session(path)
            with patch.object(measure.subprocess, "run", return_value=measure.subprocess.CompletedProcess([], 0)) as send, patch.object(measure.time, "sleep"), patch("builtins.input", return_value="") as prompt:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(measure.main(["--session", directory, "--send", "--velocities", "25", "50", "75"]), 0)
            self.assertEqual(prompt.call_count, 36)
            sent = [(c.args[0][c.args[0].index("--position")+1], c.args[0][c.args[0].index("--velocity")+1]) for c in send.call_args_list]
            expected = [(str(p), str(v)) for v in (25,50,75) for _ in range(3) for p in (0,25,75,25)]
            self.assertEqual(sent, expected)

    def test_stopped_recorder_cannot_send(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.session(path)
            with (path / "messages.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(kind="summary"))+"\n")
            with patch.object(measure.subprocess, "run") as send, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                measure.main(["--session", directory, "--send"])
            send.assert_not_called()

    def test_timing_uses_observed_travel_in_both_directions(self):
        for start, end in ((19.,69.),(69.,31.)):
            samples = []
            for tick in range(-100,501):
                t = tick / 100
                fraction = min(1, max(0, (t-.2)/1.))
                value = start+(end-start)*fraction
                samples.append((100+t,[value,value]))
            result = analyze.timing_metrics(samples,100,0)
            self.assertTrue(result["valid"])
            self.assertAlmostEqual(result["travel_percent"],end-start)
            self.assertAlmostEqual(result["travel_10_90_s"],.8,delta=.03)
            self.assertAlmostEqual(result["feedback_onset_delay_s"],.22,delta=.03)
            self.assertAlmostEqual(result["end_feedback_percent"],end)
        self.assertFalse(analyze.timing_metrics([],100,0)["valid"])

    def test_recorder_stops_between_steps_no_further_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.session(path)
            def stop_recorder(*_):
                with (path / "messages.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(kind="summary"))+"\n")
            with patch.object(measure.subprocess, "run", return_value=measure.subprocess.CompletedProcess([],0)) as send, patch.object(measure.time,"sleep",side_effect=stop_recorder), patch("builtins.input",return_value=""):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(measure.main(["--session",directory,"--send"]),1)
            self.assertEqual(send.call_count,1)

    def test_filter_does_not_discard_real_zero_position(self):
        self.assertTrue(analyze.known_synthetic(dict(position=[0,0],velocity=[0,0],effort=[0,0],state=[2,2])))
        self.assertFalse(analyze.known_synthetic(dict(position=[0,0],velocity=[0,0],effort=[.1,0],state=[2,2])))
        self.assertFalse(analyze.known_synthetic(dict(position=[0,0],velocity=[0,0],effort=[0,0],state=[1,1])))
