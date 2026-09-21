#!/usr/bin/env python3
"""Read-only SSH wrist-camera sampling: timestamp audit and one image per hand."""
import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REMOTE = r'''
import base64,json,statistics,threading,time
import rospy
from sensor_msgs.msg import CompressedImage,CameraInfo
rospy.init_node('gripper_camera_probe',anonymous=True,disable_signals=True)
lock=threading.Lock()
data={side:dict(times=[],receipts=[],ages=[],snapshot=None,info=None) for side in ('left','right')}
def image(msg,side):
    now=time.time()
    with lock:
        row=data[side];stamp=msg.header.stamp.to_sec()
        row['times'].append(stamp);row['receipts'].append(now);row['ages'].append(now-stamp)
        if row['snapshot'] is None:
            row['snapshot']=msg
def info(msg,side):
    with lock:
        data[side]['info']=dict(width=msg.width,height=msg.height,K=list(msg.K),D=list(msg.D),
                               distortion_model=msg.distortion_model,frame_id=msg.header.frame_id)
subs=[]
for side in data:
    subs.extend([rospy.Subscriber('/'+side+'_wrist_camera/color/image_raw/compressed',CompressedImage,image,callback_args=side,queue_size=4),
                 rospy.Subscriber('/'+side+'_wrist_camera/color/camera_info',CameraInfo,info,callback_args=side,queue_size=4)])
time.sleep(float(__import__('sys').argv[1]))
for sub in subs:sub.unregister()
with lock:
    for side,row in data.items():
        stamps=row['times'];receipts=row['receipts'];unique=sorted(set(stamps))
        deltas=[b-a for a,b in zip(stamps,stamps[1:])]
        snap=row['snapshot']
        print(json.dumps(dict(kind='camera_probe',side=side,
            topic='/'+side+'_wrist_camera/color/image_raw/compressed',count=len(stamps),unique_stamps=len(unique),
            header_fps=(len(unique)-1)/(unique[-1]-unique[0]) if len(unique)>1 else None,
            receipt_fps=(len(receipts)-1)/(receipts[-1]-receipts[0]) if len(receipts)>1 else None,
            median_header_age_s=statistics.median(row['ages']) if row['ages'] else None,
            min_header_age_s=min(row['ages']) if row['ages'] else None,
            max_header_gap_s=max(deltas) if deltas else None,nonincreasing_header_count=sum(d<=0 for d in deltas),
            camera_info=row['info'],snapshot_header_unix_s=snap.header.stamp.to_sec() if snap else None,
            snapshot_receipt_unix_s=receipts[0] if receipts else None,
            frame_id=snap.header.frame_id if snap else None,format=snap.format if snap else None,
            image_base64=base64.b64encode(snap.data).decode('ascii') if snap else None)),flush=True)
rospy.signal_shutdown('probe finished')
'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default=os.environ.get('KUAVO_ROBOT_SSH'))
    parser.add_argument('--workspace',default=os.environ.get('KUAVO_ROBOT_WORKSPACE'))
    parser.add_argument('--ros-master',default='http://kuavo_master:11311')
    parser.add_argument('--duration',type=float,default=6)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not 1<=args.duration<=20 or not args.host or args.host.startswith('-'):
        parser.error('Duration must be 1–20 s; set --host or KUAVO_ROBOT_SSH')
    if not args.workspace:
        parser.error('Set --workspace or KUAVO_ROBOT_WORKSPACE')
    if args.output.exists():parser.error('Choose a new output folder')
    remote='set -e; source /opt/ros/noetic/setup.bash; source '+shlex.quote(args.workspace+'/devel/setup.bash')
    remote+='; export ROS_MASTER_URI='+shlex.quote(args.ros_master)+'; exec python3 -u -c '+shlex.quote(REMOTE)+' '+shlex.quote(str(args.duration))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.host,'bash -c '+shlex.quote(remote)],stdout=subprocess.PIPE,text=True,timeout=45)
    if result.returncode:return result.returncode
    args.output.mkdir(parents=True)
    rows=[]
    for line in result.stdout.splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        payload=row.pop('image_base64',None)
        if payload:
            extension='png' if 'png' in (row.get('format') or '').lower() else 'jpg'
            path=args.output/(row['side']+'_wrist.'+extension)
            path.write_bytes(base64.b64decode(payload))
            row['snapshot_file']=path.name
        rows.append(row)
    (args.output/'camera_probe.json').write_text(json.dumps(dict(host=args.host,duration_s=args.duration,cameras=rows),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),cameras=rows),indent=2))
    return 0 if len(rows)==2 and all(r['count']>1 and r.get('snapshot_file') for r in rows) else 1


if __name__=='__main__':sys.exit(main())
