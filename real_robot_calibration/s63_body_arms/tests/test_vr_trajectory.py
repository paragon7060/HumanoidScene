"""Replay candidates remain offline; verify mapping, bias exclusion and bounds."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]))
from trajectory import NAMES, validate_plan
from vr_trajectory import candidate, validate_candidate


class VRExportTests(unittest.TestCase):
    def source(self,folder,reset=False,gap=False,large=False,closed=True):
        names=list(reversed(NAMES));index=names.index('zarm_l5_joint')
        rows=[{'kind':'metadata','schema':'kuavo_joint_response_v1','robot_model':'s63',
               'logical_target_unbiased':True,'joint_names':names}]
        for i in range(61):
            q=[0.]*18;q[index]=math.radians((2 if large else .5)*i/60)
            rows.append({'kind':'step','segment':1,'seq':i+(1 if gap and i>30 else 0),
                'sim_start_s':i/30,'sim_end_s':(i+1)/30,'exportable':not(reset and i==30),
                'reset_after_step':reset and i==30,'collision':False,'logical_joint_target_rad':q,
                'solver_joint_target_rad':[v+1 for v in q],
                'root_before_w':[0,0,0,1,0,0,0],'root_after_w':[0,0,0,1,0,0,0]})
        if closed:rows.append({'kind':'closed','ok':True})
        path=Path(folder)/'source.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        capture={'robot_version':63,'joint_names':NAMES,'baseline_source':'live_quick_mode_targets',
                 'baseline_rad':[.01]*18}
        return path,capture,{n:(-2,2) for n in NAMES}

    def test_mapping_and_solver_bias_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            path,c,l=self.source(d);p=candidate(path,c,l,1,'zarm_l5_joint')
        self.assertFalse(p['send_supported'])
        self.assertAlmostEqual(p['max_offset_deg'],.5)
        self.assertLessEqual(p['max_velocity_deg_s'],.469+1e-9)
        self.assertLessEqual(p['max_acceleration_deg_s2'],.361+1e-9)
        self.assertAlmostEqual(p['phases'][-3]['to_rad'][8],.01+math.radians(.5))
        self.assertEqual(p['phases'][-1]['to_rad'],c['baseline_rad'])
        with self.assertRaises(ValueError):validate_plan(p)

    def test_tracking_reset_and_missing_sample_rejected(self):
        for option in ('reset','gap'):
            with tempfile.TemporaryDirectory() as d:
                path,c,l=self.source(d,**{option:True})
                with self.assertRaises(ValueError):candidate(path,c,l,1,'zarm_l5_joint')

    def test_large_motion_requires_explicit_scaling(self):
        with tempfile.TemporaryDirectory() as d:
            path,c,l=self.source(d,large=True)
            with self.assertRaises(ValueError):candidate(path,c,l,1,'zarm_l5_joint')
            p=candidate(path,c,l,1,'zarm_l5_joint',scale=.25)
        self.assertAlmostEqual(p['max_offset_deg'],.5)

    def test_incomplete_log_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,c,l=self.source(d,closed=False)
            with self.assertRaises(ValueError):candidate(path,c,l,1,'zarm_l5_joint')

    def test_edited_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,c,l=self.source(d);p=candidate(path,c,l,1,'zarm_l5_joint')
            self.assertEqual(validate_candidate(p,l),p)
            p['phases'][1]['to_rad'][8]+=1
            with self.assertRaises(ValueError):validate_candidate(p,l)

    def test_joint_limit_margin_checked_after_real_rebase(self):
        with tempfile.TemporaryDirectory() as d:
            path,c,l=self.source(d);c['baseline_rad'][8]=2-math.radians(.6)
            with self.assertRaises(ValueError):candidate(path,c,l,1,'zarm_l5_joint')


if __name__=='__main__':unittest.main()
