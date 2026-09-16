#!/usr/bin/env python3
"""Audit minor-loop widths, command order and hardware-tail feedback offline.

Writes reports only; does not alter runtime mapping or send robot commands.
"""
import argparse
from bisect import bisect_right
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kuavo_isaaclab_scene.robots.gripper_action import DirectionalGripperMapping
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.twofinger_geometry import TwoFingerGeometry


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=folder / "data/s63_gripper_reversal_01")
    parser.add_argument("--uncertainty-mm", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.uncertainty_mm) or args.uncertainty_mm <= 0:
        parser.error("Uncertainty must be positive and finite")
    raw_width = json_rows(args.session / "reversal_measurements.jsonl")
    measurements = json.loads((folder / "measured_parameters.json").read_text())
    same_reference_confirmed = measurements.get("reversal_calibration_draft", {}).get("same_measurement_reference_confirmed_by_user", False)
    corrected = measurements.get("closing_75_correction", {}).get("adopted_width_mm") == 26
    if raw_width[-1]["kind"] != "completed":
        raise ValueError("Measurement protocol is incomplete")
    observations = [row for row in raw_width if row["kind"] == "width_measurement"]
    command_log = json_rows(args.session / "commands.jsonl")
    commands = [row for row in command_log if row["kind"] == "command_attempt"]
    responses = [row for row in command_log if row["kind"] == "service_response"]
    if len(commands) != len(observations) or len(responses) != len(commands) or any(not r["success"] for r in responses):
        raise ValueError("Widths and successful commands do not align one-to-one")
    for command, observation in zip(commands, observations):
        if command["position"] != [observation["command_percent"]] * 2 or command["velocity"] != [25] * 2 or command["effort"] != [1] * 2:
            raise ValueError("Unexpected command or changed measurement conditions")
    times = [row["receipt_unix_s"] for row in commands]
    tails = [deque() for _ in commands]
    counts = {"raw_state": 0, "excluded_known_synthetic_signature": 0, "retained_state": 0}
    digest = hashlib.sha256()
    for line in (args.session / "messages.jsonl").open("rb"):
        digest.update(line)
        row = json.loads(line)
        if row.get("topic") != "/leju_claw_state":
            continue
        counts["raw_state"] += 1
        # Session-specific source-backed signature; not a general zero-state rule.
        if row["position"] == [0,0] and row["velocity"] == [0,0] and row["effort"] == [0,0] and row["state"] == [2,2]:
            counts["excluded_known_synthetic_signature"] += 1
            continue
        counts["retained_state"] += 1
        stamp = row["receipt_unix_s"]
        index = bisect_right(times, stamp) - 1
        if index >= 0:
            tail = tails[index]
            tail.append((stamp, row["position"]))
            while tail and tail[0][0] < stamp - 2:
                tail.popleft()
    settings = load_gripper_settings("leju-twofinger")
    geometry = {side: TwoFingerGeometry(side) for side in ("left", "right")}
    mapper = {side: DirectionalGripperMapping(settings.sides[side].position_mapping, 1, "cpu") for side in geometry}
    comparison = []
    for command, observation, tail in zip(commands, observations, tails):
        result = dict(observation)
        result["command_receipt_unix_s"] = command["receipt_unix_s"]
        result["settled_feedback_tail_count"] = len(tail)
        result["settled_feedback_tail_median_percent"] = np.median([p for _,p in tail], axis=0).tolist() if tail else None
        for side in geometry:
            fraction = float(mapper[side].process(torch.tensor([[1-observation["command_percent"]/50]]))[0,0])
            opened = settings.command_for(side, settings.open_command)[f"{side[0]}_f_bar_1_joint"]
            predicted = geometry[side].gap_m(opened*(1-fraction))*1000
            result[f"{side}_current_mapping_width_mm"] = predicted
            measured = observation[f"{side}_width_mm"]
            result[f"{side}_mapping_error_mm"] = None if measured is None else predicted-measured
            # Conservative bounded sum; not a standard deviation or confidence interval.
            result[f"{side}_within_combined_4mm_band"] = None if measured is None else abs(predicted-measured) <= 2*args.uncertainty_mm + .03
        comparison.append(result)
    report = {"session": args.session.name, "message_sha256": digest.hexdigest(),
              "reported_measurement_error_mm": "User: about 1–2 mm; treated conservatively as ±2 mm",
              "current_uncertainty_bound_mm": args.uncertainty_mm,
              "previous_fit_uncertainty_assumption_mm": args.uncertainty_mm,
              "combined_difference_bound_mm": 2*args.uncertainty_mm,
              "cad_lookup_tolerance_mm": .03,
              "uncertainty_interpretation": "Bounded sum of measurement errors, not a statistical confidence interval",
              "feedback_window": "Last 2 seconds of retained hardware-signature messages in each command interval; not exact width-reading instant",
              "filter_source": "Previously inspected HumanoidVisualizer.cpp synthetic all-zero position/velocity/effort with state [2,2]; this session only",
              "counts": counts, "successful_commands": len(commands), "comparison": comparison,
              "finding": "Small reversals support a hold region; closing after 50-opening gives 26 mm at 75 versus 38 mm previous full-closing reference",
              "mapping_changed": False,
              "closing_75_correction_applied":corrected,
              "measurement_reference_confirmed_by_user": same_reference_confirmed,
              "pending": None if corrected else "User requested repeated command-75 comparison before fitting; near-equal feedback and 12 mm width difference remain unresolved"}
    reports = folder / "reports"
    (reports / "reversal_01_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    lines = ["# S63 중간 방향 전환 측정", "", "## 측정 오차 처리", "",
             "사용자가 알려준 1~2 mm 오차를 현재 측정의 ±2 mm로 취급한다. 이전 자 측정에도 ±2 mm를 가정하면 두 측정 비교의 보수적 차이 범위는 ±4 mm다. 표준편차나 통계적 신뢰구간으로 해석하지 않는다. 이 범위 안의 작은 차이로 매핑을 재조정하지 않는다.", "",
             "| 경로 | 명령 % | 왼손 mm | 오른손 mm | 현재 매핑 mm(왼손) | 매핑−실측 mm(왼손) | 위치 피드백 좌/우 % |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in comparison:
        median = row["settled_feedback_tail_median_percent"]
        feedback = "미확인" if median is None else f"{median[0]:.3f}/{median[1]:.3f}"
        error = row['left_mapping_error_mm']
        lines.append(f"| {row['case'][0]} | {row['command_percent']} | {row['left_width_mm']} | {row['right_width_mm']} | {row['left_current_mapping_width_mm']:.3f} | {'미측정' if error is None else f'{error:+.3f}'} | {feedback} |")
    lines += ["", "## 판정", "",
              "- A의 50→45→40에서 좌 폭 50/51/51 mm, 우 폭 52/55/53 mm다. 이 차이는 두 자 측정의 ±4 mm 비교 범위 안이며 하드웨어 피드백은 거의 유지된다.",
              "- B의 50→55→60에서 좌 폭 40/39/38 mm, 우 폭 41/39/39 mm다. 역시 작은 변화는 측정 오차 범위 안이고 위치 피드백도 거의 유지된다.",
              "- A의 25에서 좌 65 mm는 이전 열기 기준 68 mm와 3 mm 차이다. 오차 범위의 작은 차이로 보고 기존 열기 곡선을 유지한다.",
              "- B의 75에서 양손 26 mm는 최초 닫기 곡선 38 mm와 12 mm 차이였다. 이전 왼손 위치 피드백 69.048%와 이번 약 69.256%는 거의 같아, 원인을 방향 이력이나 기계 백래시로 단정할 수 없었다.",
              ("- 이후 양손의 전체 닫기/역전 경로를 각각 3회 재측정한 결과 모두 25~27 mm로 반복됐다. 현재는 대표값 26 mm로 닫기 75 노드를 보정하고 원본 38 mm는 기록으로 보존한다. 별도 전환 곡선은 적용하지 않는다. [재측정 결과](check75_01.md)." if corrected else "- 같은 측정 면을 사용자 확인했으나 원본 38 mm와 26 mm 차이는 재측정 전까지 보정을 보류한다."), "",
              f"명령 {len(commands)}개 모두 양손 동일 position, velocity 25, effort 1 A, 서비스 success를 확인했다. 원본 상태 {counts['raw_state']}개 중 알려진 합성 서명 {counts['excluded_known_synthetic_signature']}개를 이 분석에서만 제외했다. 모든 0 상태를 일반적으로 제외하는 규칙은 아니다.", "",
              "[상세 JSON](reversal_01_summary.json). 실물 명령 전송 없이 원본 로그를 분석했다."]
    (reports / "reversal_01.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report": str(reports / "reversal_01.md"), "mapping_changed":False, "counts":counts}))


if __name__ == "__main__":
    main()
