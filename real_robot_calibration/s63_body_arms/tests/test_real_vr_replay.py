"""Real VR logs convert to immutable, time-aligned sim replay plans."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from real_vr_replay import export, sample, sample_command, validate


class RealVRReplayTests(unittest.TestCase):
    def source(self, folder, gap=False):
        path = Path(folder)/"real.jsonl"
        rows = [{"kind":"metadata", "robot_version":63, "receipt_monotonic_s":10.,
                 "configuration":{"NUM_JOINT":20,"NUM_ARM_JOINT":14,
                 "min_joint_position_limits":[-180.]*20,"max_joint_position_limits":[180.]*20}}]
        for i in range(121):
            stamp=100.+i*.04+(0.2 if gap and i>=60 else 0)
            for topic, offset in (("/joint_cmd",0.),("/sensors_data_raw",-.01)):
                row={"kind":"message","topic":topic,"receipt_monotonic_s":10.+i*.04,
                    "header_s":stamp,"joint_q":[i*.001+offset]*18+[0,0],"joint_v":[.025]*18+[0,0]}
                if topic == "/joint_cmd":
                    row.update(tau=[1.5]*20,joint_kp=[10.]*20,joint_kd=[1.]*20,
                               tau_max=[18.]*20,tau_ratio=[1.]*20,control_modes=[2]*20)
                rows.append(row)
        rows.append({"kind":"summary","ok":True})
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        return path

    def test_export_preserves_real_reference_and_holds(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.source(d);plan=export(path,0,4.8,30,1)
            self.assertEqual(validate(plan),plan)
            self.assertEqual(plan["source_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            q,v,real_q,_,_,phase=sample(plan,0)
            self.assertEqual(phase,"initial_hold");self.assertEqual(v,[0.]*18)
            self.assertAlmostEqual(q[0]-real_q[0],.01)
            self.assertEqual(sample(plan,1.5)[-1],"source")
            full=sample_command(plan,1.5)
            self.assertEqual(full["target_tau"],[1.5]*18)
            self.assertEqual(full["control_modes"],[2]*18)

    def test_gap_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):export(self.source(d,True),0,4.8)
        with tempfile.TemporaryDirectory() as d:
            path=self.source(d);plan=export(path,0,4.8);path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError):validate(plan)


if __name__ == "__main__": unittest.main()
