import base64
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('record_real_gripper',Path(__file__).parents[1]/'record_real_gripper.py')
record=importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class CameraRecordingTests(unittest.TestCase):
    def row(self,format='rgb8; jpeg compressed bgr8'):
        return dict(kind='camera_frame',side='left',header_secs=100,header_nsecs=123,header_seq=5,
                    receipt_unix_s=100.02,receipt_monotonic_s=12.,format=format,
                    encoded_bytes=4,image_base64=base64.b64encode(b'abcd').decode())

    def test_original_bytes_and_timestamps_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            result=record.save_camera_frame(self.row(),folder)
            self.assertEqual((folder/result['file']).read_bytes(),b'abcd')
            self.assertEqual(result['receipt_unix_s'],100.02)
            self.assertEqual(result['header_nsecs'],123)
            self.assertNotIn('image_base64',result)
            self.assertTrue(result['file'].endswith('.jpg'))

    def test_duplicate_header_frames_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            first=record.save_camera_frame(self.row('png'),folder)
            second=record.save_camera_frame(self.row('png'),folder)
            self.assertNotEqual(first['file'],second['file'])
            self.assertTrue(first['file'].endswith('.png'))
            self.assertEqual((folder/first['file']).read_bytes(),(folder/second['file']).read_bytes())

    def test_invalid_sides_do_not_connect_or_create_session(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'session'
            with patch.object(record.subprocess,'Popen') as connect, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                record.main(['--output',str(output),'--camera-sides','left','left'])
            connect.assert_not_called()
            self.assertFalse(output.exists())

    def test_state_rows_unchanged_and_corrupt_length_rejected(self):
        state=dict(kind='message',topic='/leju_claw_state',position=[0,0])
        self.assertIs(record.save_camera_frame(state,Path('/unused')),state)
        with tempfile.TemporaryDirectory() as directory:
            row=self.row();row['encoded_bytes']=10
            with self.assertRaises(ValueError):record.save_camera_frame(row,Path(directory))
