"""Static-log quality tests; no ROS, SSH or real commands."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]))
from static_pose import analyze


class StaticPoseTests(unittest.TestCase):
    def sample(self,folder,moving=False,frozen=False,missing_motor=False,effort_nan=False):
        rows=[{"kind":"message","topic":"/enable_control_state","data":True}]
        for i in range(150):
            stamp=9+i*.02
            q=[0.]*20
            if moving:q[4]=i*.00003
            common={"kind":"message","receipt_monotonic_s":stamp,
                    "header_s":0 if frozen else stamp,"publisher":"/main"}
            rows.append(dict(common,topic="/sensors_data_raw",joint_q=q,joint_v=[0.]*20,
                             joint_torque=[float('nan') if effort_nan else 1.]*20))
            if not missing_motor:
                rows.append(dict(common,topic="/joint_cmd",joint_q=[0.]*20,control_modes=[2]*20,
                                 joint_kp=[1.]*20,joint_kd=[2.]*20,tau=[2.]*20))
            rows.append(dict(common,topic="/imu",imu_quat_xyzw=[0.,0.,0.,1.],
                             frame_id="imu",orientation_covariance=[0.]*9))
        rows.append({"kind":"summary","ok":True,"receipt_monotonic_s":12.})
        path=Path(folder)/"record.jsonl"
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return analyze(path)

    def test_quiet_valid_record(self):
        with tempfile.TemporaryDirectory() as d:r=self.sample(d)
        self.assertTrue(r['valid_for_static_comparison'])
        self.assertGreaterEqual(r['paired_samples'],60)
        self.assertEqual(r['joints']['zarm_l1_joint']['measured_effort_raw_mean'],1.)

    def test_actual_movement_rejected_despite_zero_reported_velocity(self):
        with tempfile.TemporaryDirectory() as d:r=self.sample(d,moving=True)
        self.assertFalse(r['valid_for_static_comparison'])
        self.assertIn('zarm_l1_joint: measured posture not quiet',r['failures'])

    def test_frozen_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:r=self.sample(d,frozen=True)
        self.assertFalse(r['valid_for_static_comparison'])

    def test_missing_motor_rejected(self):
        with tempfile.TemporaryDirectory() as d:r=self.sample(d,missing_motor=True)
        self.assertFalse(r['valid_for_static_comparison'])

    def test_unusable_effort_does_not_invalidate_position_comparison(self):
        with tempfile.TemporaryDirectory() as d:r=self.sample(d,effort_nan=True)
        self.assertTrue(r['valid_for_static_comparison'])
        self.assertIsNone(r['joints']['zarm_l1_joint']['measured_effort_raw_mean'])


if __name__=='__main__':
    unittest.main()
