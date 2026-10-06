#!/usr/bin/env python3
"""Export closed CPU workplace movies with an accurate TRAIN caption.

Original movies and metadata stay immutable. Only the existing top title bar
changes; the measured body poses, terminal frame, timing and status line remain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from browser_video import encode_browser_video
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results


def caption_closed_video(source, destination, title):
    import cv2
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    capture=cv2.VideoCapture(str(source))
    fps=capture.get(cv2.CAP_PROP_FPS);expected=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps<=0 or expected<1 or height<66:
        capture.release();raise ValueError('A complete measured movie with its existing title bar is required')
    pending=destination.with_name('.'+uuid.uuid4().hex+'_caption_raw.mp4')
    writer=cv2.VideoWriter(str(pending),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():capture.release();raise RuntimeError('Could not create caption export')
    count=0;last=None
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            cv2.rectangle(frame,(0,0),(width-1,31),(23,28,36),-1)
            cv2.putText(frame,title,(15,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
            writer.write(frame);count+=1;last=frame
        writer.release();capture.release()
        if count!=expected:raise ValueError('Caption export must retain every original frame')
        encoding=encode_browser_video(pending,destination)
        check=cv2.VideoCapture(str(destination))
        try:
            if int(check.get(cv2.CAP_PROP_FRAME_COUNT))!=expected or abs(check.get(cv2.CAP_PROP_FPS)-fps)>1e-6:
                raise ValueError('Caption export changed evidence frame count or timing')
        finally:check.release()
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Original movie changed during caption export')
        if not cv2.imwrite(str(destination.with_suffix('.png')),last):
            raise RuntimeError('Could not save the measured final-frame preview')
        return dict(source_SHA256=digest,frames=count,fps=fps,browser_encoding=encoding,
            existing_title_bar_only_changed=True,physics_not_replayed=True,
            original_video_unchanged=True,output_SHA256=hashlib.sha256(destination.read_bytes()).hexdigest())
    finally:
        writer.release();capture.release();pending.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.experiment_dir.resolve()
    if parent.stat().st_uid!=os.getuid():parser.error('Only an owned managed experiment may be exported')
    status=json.loads((parent/'status.json').read_text())
    if Path('/proc',str(status['training_pid'])).exists() or status.get('training_exit_code')!=0:
        parser.error('Original writer must stop normally before export')
    run=Path(json.loads((parent/'launch.json').read_text())['run']).resolve()
    if run.parent!=parent:parser.error('Run must belong to this managed experiment')
    manifest=json.loads((run/'manifest.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
    results=summarize_workplace_results(manifest,metrics)
    if args.output_dir.exists():parser.error('A unique export directory is required')
    args.output_dir.mkdir(parents=True);exports=[]
    for proof in sorted(run.glob('eval_wave_*_env_*_h264.json')):
        record=json.loads(proof.read_text());i=record['environment'];case=results['cases'][i]
        if record.get('source_video_writer_closed') is not True or record.get('actual_outcome')!=case['actual_terminal']:
            raise ValueError('Movie proof must match the final original TRAIN outcome')
        filename=record['browser_file']
        if Path(filename).name!=filename:raise ValueError('Movie must remain inside the source run')
        title=f'CPU PhysX | frozen TRAIN | env{i} {case["region"]} {case["candidate"]} | actor0 Q0'
        destination=args.output_dir/filename
        encoding=caption_closed_video(run/filename,destination,title)
        corrected=dict(recorded_split='train',role='frozen_CPU_TRAIN_workplace_probe_NOT_DEV_score',
            source_video_writer_closed=True,browser_file=filename,
            actor_updates=0,critic_updates=0,candidate=case['candidate'],original_case=case,
            original_metadata_role=record.get('role'),original_metadata_preserved=True,
            source_metadata_SHA256=hashlib.sha256(proof.read_bytes()).hexdigest(),**encoding)
        destination.with_suffix('.json').write_text(json.dumps(corrected,indent=2)+'\n')
        exports.append(dict(environment=i,file=filename,**encoding))
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(exports=exports,
        unique_TRAIN_cases=16,candidate_requests=128,not_independent_DEV_score=True,
        all178_tensors_and_actor_Q_replay_counters_frozen=True,goal_not_complete=True),indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),closed_videos_exported=len(exports))))


if __name__=='__main__':main()
