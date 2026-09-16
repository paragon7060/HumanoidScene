#!/usr/bin/env python3
"""Analyze recorded hardware-feedback timing without connecting to the robot."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from measure_dynamics import make_plan


def known_synthetic(row):
    return (row.get("position") == [0, 0] and row.get("velocity") == [0, 0]
            and row.get("effort") == [0, 0] and row.get("state") == [2, 2])


def timing_metrics(samples, command_time, side_index):
    """Times use hardware feedback, not optical jaw motion; no nominal-% normalization."""
    before = [(t, p[side_index]) for t, p in samples if command_time - 1 <= t < command_time]
    after = [(t - command_time, p[side_index]) for t, p in samples if command_time <= t <= command_time + 5]
    if len(before) < 20 or not after or before[-1][0] - before[0][0] < .5 or after[-1][0] < 4.9:
        return {"valid": False, "reason": "insufficient baseline or 5-second feedback window"}
    gaps = [after[i][0] - after[i-1][0] for i in range(1, len(after))]
    if max(gaps, default=0) > .1:
        return {"valid": False, "reason": "feedback gap exceeds 100 ms"}
    start = median(v for _, v in before)
    tail = [v for t, v in after if t >= 4]
    end = median(tail)
    travel = end - start
    base_range = max(v for _, v in before) - min(v for _, v in before)
    result = dict(valid=True, start_feedback_percent=start, end_feedback_percent=end,
                  travel_percent=travel, baseline_range_percent=base_range,
                  tail_range_percent=max(tail)-min(tail), max_sample_gap_s=max(gaps, default=0))
    if abs(travel) < 5:
        return dict(result, valid=False, reason="feedback travel below 5 percentage points")
    if base_range > 2:
        return dict(result, valid=False, reason="baseline not stationary (range above 2 percentage points)")
    # Causal 20-ms median reduces isolated packet noise; this introduces smoothing delay.
    smoothed = []
    left = 0
    for i, (t, _) in enumerate(after):
        while after[left][0] < t - .02:
            left += 1
        smoothed.append((t, (median(v for _, v in after[left:i+1]) - start) / travel))
    def crossing(level):
        previous = (0., 0.)
        for t, fraction in smoothed:
            if fraction >= level:
                pt, pf = previous
                return pt + (t-pt)*(level-pf)/(fraction-pf) if fraction > pf else t
            previous = (t, fraction)
        return None
    t10, t90 = crossing(.1), crossing(.9)
    # Onset: at least 0.5 percentage point or 2% travel, sustained for 50 ms.
    onset_level = max(.5 / abs(travel), .02)
    onset_start = None
    onset = None
    for t, fraction in smoothed:
        if fraction >= onset_level:
            if onset_start is None:
                onset_start = t
            if t - onset_start >= .05:
                onset = onset_start
                break
        else:
            onset_start = None
    # Settling is the last exit from a +/-2 percentage-point terminal band,
    # followed by >=0.5 s observed inside it; not physical width settling.
    outside = [i for i, (_, fraction) in enumerate(smoothed) if abs(fraction-1)*abs(travel) > 2]
    settle_i = outside[-1]+1 if outside else 0
    settle = smoothed[settle_i][0] if settle_i < len(smoothed) and smoothed[-1][0]-smoothed[settle_i][0] >= .5 else None
    result.update(feedback_onset_delay_s=onset, t10_s=t10, t90_s=t90,
                  travel_10_90_s=t90-t10 if t10 is not None and t90 is not None else None,
                  feedback_settle_s=settle,
                  overshoot_percent=max(0, max(f for _, f in smoothed)-1)*abs(travel))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args(argv)
    read = lambda name: [json.loads(line) for line in (args.session / name).read_text().splitlines()]
    records = read("dynamics_measurements.jsonl")
    protocol = records[0]
    if protocol.get("protocol") != "dynamics" or records[-1]["kind"] != "completed":
        parser.error("완료된 dynamics 세션이 필요합니다. 중단된 원본 로그는 유지하세요")
    plan = make_plan(protocol["velocities"], protocol["repeats"])
    attempts = [r for r in records if r["kind"] == "step_attempt"]
    holds = [r for r in records if r["kind"] == "hold_completed"]
    keys = ("velocity", "trial", "step", "command_percent", "measured_transition")
    project = lambda rows: [tuple(r[k] for k in keys) for r in rows]
    if project(attempts) != project(plan) or project(holds) != project(plan) or protocol.get("effort_A") != 1 or protocol.get("hold_s") != 5:
        parser.error("측정 계획/전송/정지 기록이 일치하지 않습니다")
    log = read("commands.jsonl")
    commands = [r for r in log if r["kind"] == "command_attempt"]
    responses = [r for r in log if r["kind"] == "service_response"]
    if len(commands) != len(plan) or len(responses) != len(plan) or not all(r["success"] for r in responses):
        parser.error("명령 수 또는 서비스 성공 기록을 확인하세요")
    for command, row in zip(commands, plan):
        if command["position"] != [row["command_percent"]]*2 or command["velocity"] != [row["velocity"]]*2 or command["effort"] != [1]*2:
            parser.error("계획과 실제 명령이 다릅니다")
    times = [r["receipt_unix_s"] for r in commands]
    if any(b <= a for a, b in zip(times, times[1:])):
        parser.error("명령 시계가 역전되었거나 순서가 잘못되었습니다")
    windows = [[] for _ in commands]
    counts = dict(raw_state=0, excluded_known_synthetic_signature=0, retained_state=0, malformed_state=0)
    digest = hashlib.sha256()
    with (args.session / "messages.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            r = json.loads(line)
            if r.get("topic") != "/leju_claw_state":
                continue
            counts["raw_state"] += 1
            if known_synthetic(r):
                counts["excluded_known_synthetic_signature"] += 1
                continue
            try:
                p = [float(r["position"][r["name"].index(name)]) for name in ("left_claw", "right_claw")]
                t = float(r["receipt_unix_s"])
                if not all(math.isfinite(v) for v in [t, *p]):
                    raise ValueError("nonfinite")
            except (KeyError, IndexError, ValueError, TypeError):
                counts["malformed_state"] += 1
                continue
            counts["retained_state"] += 1
            index = bisect_right(times, t)-1
            for target in (index, index+1):
                if 0 <= target < len(times) and plan[target]["measured_transition"] and times[target]-1 <= t <= times[target]+5:
                    windows[target].append((t, p))
    observations = []
    for row, command_time, samples in zip(plan, times, windows):
        if not row["measured_transition"]:
            continue
        samples.sort(key=lambda item: item[0])
        for side_index, side in enumerate(("left", "right")):
            observations.append(dict(**row, side=side, command_receipt_unix_s=command_time,
                                     direction="closing" if row["step"] == 3 else "opening",
                                     **timing_metrics(samples, command_time, side_index)))
    groups = defaultdict(list)
    for row in observations:
        if row["valid"]:
            groups[(row["velocity"], row["side"], row["direction"])].append(row)
    grouped = []
    metrics = ("feedback_onset_delay_s", "travel_10_90_s", "feedback_settle_s", "end_feedback_percent")
    for (velocity, side, direction), rows in groups.items():
        result = dict(velocity=velocity, side=side, direction=direction, valid_trials=len(rows))
        for key in metrics:
            values = [r[key] for r in rows if r[key] is not None]
            result[key] = dict(median=median(values), min=min(values), max=max(values)) if values else None
        grouped.append(result)
    report = dict(session=args.session.name, successful_commands=len(commands), counts=counts,
                  messages_sha256=digest.hexdigest(), observations=observations, grouped=grouped,
                  scope="Hardware feedback timing only; not optical tip timing, physical width, torque or force calibration",
                  clock="Command attempt and state receipt Unix clocks on the same remote host; includes ROS/service/receipt overhead",
                  method="1 s pre-command baseline; 4-5 s terminal median; measured travel normalization; causal 20 ms median; 50 ms sustained onset; +/-2 percentage point settling band",
                  filter_source="Previously inspected S63 HumanoidVisualizer synthetic signature: exact zero position/velocity/effort and state [2,2]; not a universal zero filter")
    report_dir = Path(__file__).resolve().parent / "reports"
    target = report_dir / f"{args.session.name}_dynamics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# S63 gripper 시간 응답", "", "하드웨어 위치 피드백 기준 분석이다. 실제 끝단 이동 지연과 같다고 단정하지 않는다.", "",
             "| 속도 | 손 | 방향 | 반복 | 피드백 시작 지연 s | 10–90% s | 정착 s | 끝 위치 % |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return f"{value:.4f}" if value is not None else "미확인"
    for r in observations:
        if not r["valid"]:
            lines.append(f"| {r['velocity']} | {r['side']} | {r['direction']} | {r['trial']} | 제외: {r['reason']} | | | |")
        else:
            lines.append(f"| {r['velocity']} | {r['side']} | {r['direction']} | {r['trial']} | {fmt(r['feedback_onset_delay_s'])} | {fmt(r['travel_10_90_s'])} | {fmt(r['feedback_settle_s'])} | {fmt(r['end_feedback_percent'])} |")
    lines += ["", "원격 명령 직전 시각과 동일 호스트의 ROS 수신 시각을 비교했다. ROS/서비스 지연이 포함된다.",
              "10–90%는 명목 25/75가 아닌 관측 시작·끝 위치 차이로 정규화한다. 피드백을 폭으로 환산하지 않는다.",
              "20 ms 인과 중앙값 필터와 시작 검출(최소 0.5%p 또는 이동량 2%, 50 ms 유지)을 사용했다.",
              "정착은 끝 위치 ±2%p 안에 최소 0.5초 남아 있는 기준이다. 센서 노이즈와 프로토콜 기준이며 ±2 mm 폭 오차와 다른 단위다.",
              "알려진 시각화 합성값만 제외하고 원본을 보존했다. 영상이 있으면 실제 jaw 움직임과 대조한다.",
              f"[원본 분석 JSON]({target.name}). 이 분석은 Isaac 설정을 수정하지 않는다."]
    target.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(report=str(target), valid_observations=sum(r["valid"] for r in observations), total_observations=len(observations))))


if __name__ == "__main__":
    main()
