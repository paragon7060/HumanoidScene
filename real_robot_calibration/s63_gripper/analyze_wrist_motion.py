#!/usr/bin/env python3
"""Offline feature-motion audit for the fixed wrist views in motion_01.

Requires OpenCV and NumPy. Image coordinates are session-specific, not a generic
tip detector. No robot access; original images and timestamps remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from analyze_dynamics import known_synthetic

ROI = {
    'left': {
        'closing': [[(110,285),(280,215),(270,330),(180,400)],
                    [(653,240),(745,265),(816,331),(777,378),(676,319)]],
        'opening': [[(345,220),(402,185),(402,325),(347,325)],
                    [(515,188),(610,238),(590,322),(520,260)]],
        'background': [[(335,25),(490,25),(490,155),(335,155)]],
    },
    'right': {
        'closing': [[(110,85),(175,100),(268,200),(265,230),(170,190)],
                    [(685,145),(825,65),(835,145),(670,228)]],
        'opening': [[(329,135),(386,155),(398,268),(355,240)],
                    [(508,220),(570,165),(593,215),(510,265)]],
        'background': [[(12,315),(380,315),(380,470),(12,470)]],
    },
}


def features(image, polygon):
    mask = np.zeros(image.shape, np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    found = cv2.goodFeaturesToTrack(image, 180, .005, 4, mask=mask, blockSize=3)
    if found is None or len(found) < 6:
        raise ValueError('Insufficient textured features in audited ROI')
    return found.reshape(-1, 2)


def track(images, polygons):
    batches = [features(images[0], p) for p in polygons]
    initial = np.concatenate(batches)
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(batches)])
    current = initial.reshape(-1, 1, 2).astype(np.float32)
    paths = [initial.copy()]
    good = np.ones(len(initial), bool)
    params = dict(winSize=(25, 25), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .01))
    for previous, image in zip(images, images[1:]):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, current, None, **params)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(image, previous, nxt, None, **params)
        good &= status.ravel().astype(bool) & reverse.ravel().astype(bool)
        good &= np.linalg.norm(back.reshape(-1, 2)-current.reshape(-1, 2), axis=1) < 1.
        paths.append(nxt.reshape(-1, 2).copy())
        current = nxt
    # A fixed feature set avoids changing medians when a feature drops out.
    return np.asarray(paths)[:, good], labels[good], [len(p) for p in batches]


def analyze(images, times, side, direction):
    points, labels, initial_counts = track(images, ROI[side][direction]+ROI[side]['background'])
    bg = labels == 2
    retained = [int(np.sum(labels == i)) for i in range(3)]
    if any(n < 6 for n in retained):
        return dict(valid=False, reason='Too few persistent jaw/background features', retained=retained)
    corrected = []
    bg_residual = []
    for frame in points:
        transform, inliers = cv2.estimateAffinePartial2D(points[0, bg], frame[bg],
                                                        method=cv2.RANSAC, ransacReprojThreshold=1.)
        if transform is None or np.sum(inliers) < 6:
            return dict(valid=False, reason='Background registration failed')
        expected = points[0] @ transform[:, :2].T + transform[:, 2]
        corrected.append(frame-expected)
        bg_residual.append(float(np.median(np.linalg.norm(frame[bg]-expected[bg], axis=1))))
    corrected = np.asarray(corrected)
    baseline = (times < -.04) & (times >= -.5)
    tail = times >= 1.
    jaws = []
    for i in range(2):
        displacement = corrected[:, labels == i]
        displacement -= np.median(displacement[baseline], axis=0)
        magnitude = np.median(np.linalg.norm(displacement, axis=2), axis=1)
        noise = magnitude[baseline]
        threshold = float(max(1., np.max(noise)+.5,
                              np.median(noise)+6*np.median(abs(noise-np.median(noise)))))
        end = np.median(displacement[tail], axis=0)
        lengths2 = np.sum(end*end, axis=1)
        moving = lengths2 > 25.
        if np.sum(moving) < 6:
            return dict(valid=False, reason='Insufficient net jaw displacement')
        # Feature displacement projected onto each feature's final vector.
        # This fraction is not physical width, joint angle, or arc length.
        fraction = np.median(np.sum(displacement[:, moving]*end[moving], axis=2)/lengths2[moving], axis=1)
        onset = None
        for k in range(1, len(times)-2):
            if (times[k] >= 0 and np.all(magnitude[k:k+3] >= threshold)
                    and np.all(fraction[k:k+3] >= .02)):
                onset = [float(times[k-1]), float(times[k])]
                break
        jaws.append(dict(jaw=i+1, persistent_features=retained[i],
                         baseline_motion_max_px=float(np.max(noise)), onset_threshold_px=threshold,
                         onset_header_interval_s=onset, final_displacement_median_px=float(np.median(np.sqrt(lengths2))),
                         times_s=times.tolist(), projected_feature_fraction=fraction.tolist(),
                         displacement_px=magnitude.tolist(),
                         initial_feature_coordinates=points[0, labels == i].tolist(),
                         final_feature_coordinates=points[-1, labels == i].tolist()))
    return dict(valid=True, initial_feature_counts=initial_counts, retained_feature_counts=retained,
                max_background_registration_residual_px=max(bg_residual), jaws=jaws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    session = args.session
    if session.name != 'wrist_gripper_motion_01':
        parser.error('ROI coordinates were visually audited only for wrist_gripper_motion_01')
    report_folder = Path(__file__).parent/'reports'
    feedback = json.loads((report_folder/(session.name+'_dynamics.json')).read_text())
    frames = {'left': [], 'right': []}
    samples = []
    for line in (session/'messages.jsonl').open():
        row = json.loads(line)
        if row.get('kind') == 'camera_frame':
            row['stamp'] = row['header_secs']+row['header_nsecs']*1e-9
            frames[row['side']].append(row)
        elif row.get('topic') == '/leju_claw_state' and not known_synthetic(row):
            samples.append((row['receipt_unix_s'], row['position']))
    observations = []
    for obs in feedback['observations']:
        t = obs['command_receipt_unix_s']
        fs = [f for f in frames[obs['side']] if -.55 <= f['stamp']-t <= 1.5]
        images = [cv2.imread(str(session/f['file']), cv2.IMREAD_GRAYSCALE) for f in fs]
        if any(im is None for im in images):
            raise ValueError('Missing or undecodable frame')
        times = np.array([f['stamp']-t for f in fs])
        result = analyze(images, times, obs['side'], obs['direction'])
        result.update(side=obs['side'], direction=obs['direction'], trial=obs['trial'],
                      command_receipt_unix_s=t, feedback_onset_delay_s=obs['feedback_onset_delay_s'],
                      frames=[f['file'] for f in fs], max_header_gap_s=float(np.max(np.diff(times))))
        observations.append(result)
        print(obs['side'], obs['direction'], obs['trial'], result['valid'],
              [(j['persistent_features'],j['onset_header_interval_s']) for j in result.get('jaws', [])], flush=True)
    report = dict(session=session.name, scope='Audited jaw feature motion in RGB image coordinates; not physical gap or actuator delay',
                  clock='Camera header minus remote service command-attempt receipt; exposure timestamp and camera internal latency unverified',
                  method='Sequential pyramidal LK, forward/back error below 1 px, fixed persistent feature set; background affine compensation; onset above baseline noise, >=1 px and >=2% projected feature progress for three frames',
                  roi=ROI, messages_sha256=feedback['messages_sha256'], observations=observations)
    report['status'] = 'validated_projected_rgb_feature_motion' if all(o['valid'] for o in observations) else 'incomplete_optical_tracking'
    report['limitations'] = [
        'Detection interval brackets threshold crossing in sampled camera headers, not the very first subpixel physical motion.',
        'Features are jaw texture, not exact geometric tip edges; normalized feature motion is not physical tip gap.',
        'Camera exposure timing/internal latency unverified; no pure actuator delay inferred.',
        'Different observables in the response chart are qualitative comparisons; no runtime gains or delay changed from this recording.',
        'Left background texture is weak; registration residuals near 2 px limit precise onset and jaw asymmetry interpretation.',
        'Sender queue lag affects live delivery, not the previously captured header/receipt times; local file-write time must not be used as motion time.',
    ]
    report['grouped'] = []
    for side in ['left', 'right']:
        for direction in ['closing', 'opening']:
            oo = [o for o in observations if o['side'] == side and o['direction'] == direction and o['valid']]
            intervals = [j['onset_header_interval_s'] for o in oo for j in o['jaws'] if j['onset_header_interval_s'] is not None]
            report['grouped'].append(dict(side=side, direction=direction, valid_trials=len(oo),
                optical_detection_interval_envelope_s=[min(i[0] for i in intervals), max(i[1] for i in intervals)] if intervals else None,
                max_background_registration_residual_px=max(o['max_background_registration_residual_px'] for o in oo) if oo else None))
    report['camera_audit'] = {}
    for side, fs in frames.items():
        ts = [f['stamp'] for f in fs]
        ages = [f['receipt_unix_s']-f['stamp'] for f in fs]
        report['camera_audit'][side] = dict(frames=len(fs), header_span_s=ts[-1]-ts[0],
            median_header_receipt_age_s=median(ages), max_header_receipt_age_s=max(ages),
            max_sender_queue_lag_s=max(f['sender_queue_lag_s'] for f in fs),
            nonincreasing_headers=sum(b <= a for a, b in zip(ts, ts[1:])),
            max_header_gap_s=max(b-a for a, b in zip(ts, ts[1:])),
            all_files_length_valid=all((session/f['file']).stat().st_size == f['encoded_bytes'] for f in fs))
    output = report_folder/(session.name+'_optical.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    render_outputs(report, session, report_folder, feedback, frames, samples)


def render_outputs(report, session, folder, feedback, frames, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    sim = [json.loads(line) for line in (folder/'isaac_dynamics_calibrated_continuous.jsonl').open()]
    for row, direction in enumerate(['closing', 'opening']):
        for col, side in enumerate(['left', 'right']):
            ax = axes[row, col]
            for obs in report['observations']:
                if obs['side'] != side or obs['direction'] != direction or not obs['valid']:
                    continue
                for jaw in obs['jaws']:
                    ax.plot(jaw['times_s'], jaw['projected_feature_fraction'],
                            color=['tab:blue', 'tab:orange'][jaw['jaw']-1], alpha=.5,
                            label=f"RGB jaw {jaw['jaw']} (3 trials)" if obs['trial'] == 1 else None)
                if obs['trial'] == 1:
                    ref = next(o for o in feedback['observations'] if o['side'] == side and o['direction'] == direction and o['trial'] == 1)
                    ii = ['left', 'right'].index(side)
                    ts = [(t-obs['command_receipt_unix_s'], p[ii]) for t, p in samples if -.2 <= t-obs['command_receipt_unix_s'] <= 1.2]
                    start, end = ref['start_feedback_percent'], ref['end_feedback_percent']
                    ax.plot([t for t, _ in ts], [(p-start)/(end-start) for _, p in ts],
                            'k-', linewidth=1.5, label='Real encoder (trial 1, raw)')
            step = 3 if direction == 'closing' else 4
            sf = [s for s in sim if s['side'] == side and s['trial'] == 1 and s['step'] == step]
            command_t = sf[0]['time_s']-1/120
            previous = next(s for s in reversed(sim) if s['side'] == side and s['time_s'] <= command_t)
            start = previous['driver_feedback_fraction_percent'][0]
            end = sf[-1]['driver_feedback_fraction_percent'][0]
            sf = [s for s in sf if s['time_s']-command_t <= 1.2]
            ax.plot([s['time_s']-command_t for s in sf], [(s['driver_feedback_fraction_percent'][0]-start)/(end-start) for s in sf],
                    'g--', linewidth=2, label='Current Isaac driver (trial 1)')
            ax.axvline(0, color='gray', linewidth=.8)
            ax.set(title=f'{side.capitalize()} / {direction}', xlim=(-.1, 1.1), ylim=(-.05, 1.08))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='lower right')
            ax.set_xlabel('Seconds after command attempt / simulation target update')
            ax.set_ylabel('Normalized progress (different observables)')
    fig.suptitle('Wrist RGB feature motion, real encoder and current Isaac response')
    fig.text(.5, .015, 'RGB fractions are projected feature displacement, not tip gap or joint angle. Camera exposure timing is unverified.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(folder/'wrist_gripper_motion_01_response.png', dpi=150)
    plt.close(fig)
    # Inspect the fixed retained feature set at the start and end of trial 1.
    sheet = np.zeros((4*260, 2*424, 3), np.uint8)
    for row, (side, direction) in enumerate([(s, d) for s in ['left', 'right'] for d in ['closing', 'opening']]):
        obs = next(o for o in report['observations'] if o['side'] == side and o['direction'] == direction and o['trial'] == 1)
        if not obs['valid']:
            continue
        for col, (which, coordinate) in enumerate([(0, 'initial_feature_coordinates'), (-1, 'final_feature_coordinates')]):
            im = cv2.imread(str(session/obs['frames'][which]))
            for jaw in obs['jaws']:
                for x, y in jaw[coordinate]:
                    cv2.circle(im, (round(x), round(y)), 3, [(255,120,0),(0,180,255)][jaw['jaw']-1], -1)
            sheet[row*260:row*260+240, col*424:(col+1)*424] = cv2.resize(im, (424, 240))
            cv2.putText(sheet, f'{side} {direction} / '+('baseline' if col == 0 else 'terminal'),
                        (col*424+5, row*260+256), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    cv2.imwrite(str(folder/'wrist_gripper_motion_01_tracking.jpg'), sheet)


if __name__ == '__main__':
    main()
