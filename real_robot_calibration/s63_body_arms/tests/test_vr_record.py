"""Passive VR serialization keeps original command units and timestamps."""
import ast
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parents[1]))
from remote import WORKER


def functions(**scope):
    definitions=[n for n in ast.parse(WORKER).body if isinstance(n,ast.FunctionDef) and
                 n.name in ('ros_payload','receive_vr_input')]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'vr_callbacks','exec'),scope)
    return scope


class Stamp:
    def to_sec(self):return 123.45


class Header:
    __slots__=('stamp','seq')
    def __init__(self):self.stamp=Stamp();self.seq=9


class JointState:
    __slots__=('header','name','position','velocity','effort','_connection_header')
    def __init__(self):
        self.header=Header();self.name=['arm_joint_1'];self.position=[30.]
        self.velocity=[2.];self.effort=[];self._connection_header={'callerid':'/ik_ros_uni'}


class VRRecordTests(unittest.TestCase):
    def test_nested_ros_fields_and_device_times_preserved(self):
        class PoseList:
            __slots__=('timestamp_ms','is_high_confidence','poses')
        class Pose:
            __slots__=('position',)
        p=Pose();p.position=(1.,2.,3.)
        m=PoseList();m.timestamp_ms=987654321;m.is_high_confidence=False;m.poses=[p]
        scope=functions()
        self.assertEqual(scope['ros_payload'](m),{'timestamp_ms':987654321,'is_high_confidence':False,
            'poses':[{'position':[1.,2.,3.]}]})

    def test_raw_degree_commands_and_receipt_times_kept_with_rate_limit(self):
        clock=SimpleNamespace(monotonic=Mock(side_effect=[10.,10.005,10.02]),time=lambda:100.)
        log=Mock();counts={}
        scope=functions(time=clock,lock=threading.RLock(),vr_logged_at={},vr_counts=counts,enqueue_log=log)
        msg=JointState();spec=('/kuavo_arm_traj','sensor_msgs/JointState',.01)
        for _ in range(3):scope['receive_vr_input'](msg,spec)
        self.assertEqual(log.call_count,2)
        row=log.call_args.args[0]
        self.assertEqual(row['payload']['position'],[30.])
        self.assertEqual(row['payload']['velocity'],[2.])
        self.assertEqual(row['header_s'],123.45)
        self.assertEqual(row['receipt_monotonic_s'],10.02)
        self.assertEqual(row['publisher'],'/ik_ros_uni')
        self.assertEqual(counts['/kuavo_arm_traj'],2)


if __name__=='__main__':unittest.main()
