"""Synthetic logs only; verify static pairing and partial-protocol reporting."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]))
from analyze_screen import analyze_log


class ScreenAnalysisTests(unittest.TestCase):
    def log(self, path, partial=False, sparse_motor=False, missing_cycle=False):
        joint='zarm_l4_joint'
        phases=[('baseline_hold',0.),(joint+'/1/+1/hold',.25),
                (joint+'/1/+1/baseline',0.),(joint+'/1/-1/hold',-.25),
                (joint+'/1/-1/baseline',0.)]
        expected=[{'label':label,'duration_s':2.} for label,_ in phases]
        if missing_cycle:expected.append({'label':joint+'/2/+1/hold','duration_s':2.})
        rows=[{'kind':'local_attempt','plan':{'phases':expected}}]
        end=9. if partial else 10.
        for k,(label,offset) in enumerate(phases):
            q=[0.]*18;q[7]=math.radians(offset)
            rows.append({'kind':'target','phase':label,'position_rad':q,'receipt_monotonic_s':2.*k})
        rows.append({'kind':'target','phase':'abort_last_target' if partial else 'complete',
                     'position_rad':[0.]*18,'receipt_monotonic_s':end})
        for index in range(int(end/.02)):
            stamp=index*.02;label,offset=phases[min(4,int(stamp/2.))]
            cmd=[0.]*20;cmd[7]=math.radians(offset)
            measured=cmd[:];measured[7]+=math.radians(.04+(.02 if '/+1/hold' in label else .01 if '/-1/hold' in label else 0.))
            if not sparse_motor or index%5==0:
                rows.append({'kind':'message','topic':'/joint_cmd','joint_q':cmd,'receipt_monotonic_s':stamp})
            rows.append({'kind':'message','topic':'/sensors_data_raw','joint_q':measured,'receipt_monotonic_s':stamp+.001})
        rows.append({'kind':'error' if partial else 'summary','ok':not partial,'receipt_monotonic_s':end+.01})
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_initial_bias_is_not_mistaken_for_direction_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';self.log(p);r=analyze_log(p)['entries']['zarm_l4_joint']
        self.assertTrue(r['protocol_complete']);self.assertTrue(r['measurement_valid'])
        self.assertAlmostEqual(r['baseline_bias_deg'],.04)
        self.assertAlmostEqual(r['positive_corrected_deg'],.02)
        self.assertAlmostEqual(r['negative_corrected_deg'],.01)
        self.assertAlmostEqual(r['return_change_deg'],0.)
        self.assertEqual(r['followup_reasons'],[])

    def test_abort_during_final_baseline_is_not_full_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';self.log(p,partial=True);r=analyze_log(p)['entries']['zarm_l4_joint']
        self.assertFalse(r['protocol_complete'])
        self.assertIn('incomplete protocol',r['followup_reasons'])

    def test_constant_command_brackets_salvage_static_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';self.log(p,sparse_motor=True);report=analyze_log(p)
        r=report['entries']['zarm_l4_joint']
        self.assertTrue(r['measurement_valid'])
        self.assertGreater(report['stable_bracket_pairs'],0)
        self.assertAlmostEqual(r['positive_corrected_deg'],.02)

    def test_missing_planned_cycle_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';self.log(p,missing_cycle=True);r=analyze_log(p)['entries']['zarm_l4_joint']
        self.assertFalse(r['protocol_complete'])


if __name__=='__main__':
    unittest.main()
