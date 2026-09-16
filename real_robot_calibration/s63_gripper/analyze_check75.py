#!/usr/bin/env python3
"""Audit both 75-check approaches and their three repeats without robot access."""
from bisect import bisect_right
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent


def main():
    session = FOLDER / "data/s63_gripper_75_check_01"
    read = lambda name: [json.loads(line) for line in (session/name).read_text().splitlines()]
    records = read("reversal_measurements.jsonl")
    protocol = records[0]
    assert protocol["protocol"] == "check75" and protocol["repeats"] == 3 and records[-1]["kind"] == "completed"
    attempts = [r for r in records if r["kind"] == "step_attempt"]
    widths = {(r["trial"], r["case"], r["step"]): r for r in records if r["kind"] == "width_measurement"}
    expected = [(trial, name, step, percent) for trial in range(1,4)
                for name, values in (("C_full_closing_75_check", [0,25,50,75,100]),
                                     ("D_reversal_closing_75_check", [100,50,55,60,75,100]))
                for step, percent in enumerate(values,1)]
    assert [(r['trial'],r['case'],r['step'],r['command_percent']) for r in attempts] == expected
    log = read("commands.jsonl")
    commands = [r for r in log if r["kind"] == "command_attempt"]
    responses = [r for r in log if r["kind"] == "service_response"]
    assert len(commands) == len(responses) == 33 and all(r["success"] for r in responses)
    for command, attempt in zip(commands, attempts):
        assert command["position"] == [attempt["command_percent"]]*2
        assert command["velocity"] == [25]*2 and command["effort"] == [1]*2
    times = [r["receipt_unix_s"] for r in commands]
    tails = [deque() for _ in commands]
    counts = {"raw_state":0, "excluded_known_synthetic_signature":0, "retained_state":0}
    digest = hashlib.sha256()
    for line in (session / "messages.jsonl").open("rb"):
        digest.update(line)
        r = json.loads(line)
        if r.get("topic") != "/leju_claw_state": continue
        counts["raw_state"] += 1
        if r["position"] == [0,0] and r["velocity"] == [0,0] and r["effort"] == [0,0] and r["state"] == [2,2]:
            counts["excluded_known_synthetic_signature"] += 1
            continue
        counts["retained_state"] += 1
        index = bisect_right(times, r["receipt_unix_s"])-1
        if index >= 0:
            tail = tails[index]
            tail.append((r["receipt_unix_s"],r["position"]))
            while tail and tail[0][0] < r["receipt_unix_s"]-2: tail.popleft()
    results = []
    for command, attempt, tail in zip(commands, attempts, tails):
        key = (attempt["trial"],attempt["case"],attempt["step"])
        if key not in widths: continue
        row = dict(widths[key])
        assert row["command_percent"] == 75 and row["left_width_mm"] is not None and row["right_width_mm"] is not None
        row["command_receipt_unix_s"] = command["receipt_unix_s"]
        assert tail, "75 hardware tail unavailable"
        row["hardware_tail_median_percent"] = np.median([p for _,p in tail],axis=0).tolist()
        row["hardware_tail_count"] = len(tail)
        results.append(row)
    assert len(results) == 6
    grouped = {}
    pooled = []
    for prefix in ("C", "D"):
        grouped[prefix] = {}
        for side in ("left", "right"):
            values = [r[f"{side}_width_mm"] for r in results if r["case"].startswith(prefix)]
            assert len(values) == 3
            pooled.extend(values)
            grouped[prefix][side] = {"widths_mm":values, "median_mm":float(np.median(values)),
                                    "min_mm":min(values), "max_mm":max(values)}
    report = {"session":session.name, "successful_commands":33, "counts":counts,
              "messages_sha256":digest.hexdigest(), "observations":results, "grouped":grouped,
              "measurement_bound_mm":2, "difference_comparison_bound_mm":4,
              "uncertainty_interpretation":"Conservative bounded measuring error, not a standard deviation or confidence interval",
              "pooled_width_range_mm":[min(pooled),max(pooled)], "pooled_median_mm":float(np.median(pooled)),
              "previous_left_full_closing_width_mm":38,
              "decision":"Adopt 26 mm for command-75 closing on both sides; preserve old 38 mm observation as superseded, no distinct 50-reversal curve justified at 75",
              "scope":"Only 75 repeated on both hands; other full-cycle points still use prior left measurements/copied right mapping",
              "filter_source":"Session-specific synthetic all-zero position/velocity/effort and state [2,2], previously inspected HumanoidVisualizer.cpp; not a universal zero-state exclusion",
              "feedback_window":"Last 2 s of retained hardware-signature rows before next command; not exact physical width-reading instant"}
    target = FOLDER / "reports/check75_01_summary.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines = ["# S63 명령 75 재측정 결과", "", "양손 동일 명령, velocity 25, effort 1 A로 전체 닫기 C와 중간 역전 D를 3회씩 교대로 측정했다. 명령 33개 모두 서비스 success와 명령 순서를 확인했다.", "",
             "| 경로 | 반복 | 왼손 mm | 오른손 mm | 하드웨어 위치 좌/우 % |", "|---|---:|---:|---:|---|"]
    for r in results:
        p=r["hardware_tail_median_percent"]
        lines.append(f"| {r['case'][0]} | {r['trial']} | {r['left_width_mm']} | {r['right_width_mm']} | {p[0]:.3f}/{p[1]:.3f} |")
    lines += ["", "## 보정 판단", "", "- C(0→25→50→75): 왼손 25/26/27 mm, 오른손 25/25/26 mm.",
              "- D(100→50→55→60→75): 왼손 26/26/26 mm, 오른손 25/26/26 mm.",
              "- 전체 범위 25~27 mm, pooled median 26 mm다. 같은 손의 경로별 중앙값 차이는 최대 1 mm로 사용자의 1~2 mm 측정 오차 범위에서 구분되지 않는다. 두 측정 비교의 보수적 오차 범위는 ±4 mm이며 통계적 신뢰구간으로 해석하지 않는다.",
              "- 이번 반복 측정은 기존 38 mm를 재현하지 않았다. 그 원인을 단정하지 않고 기존 관측을 보존하되, 양손 닫기 75의 매핑 기준을 26 mm로 교체한다. 다른 매핑 노드는 유지한다.",
              "- 별도의 50 역전 곡선은 적용하지 않는다. 55/60에서의 작은 변화는 기존 유지 규칙과 측정 오차로 설명 가능한 범위다.",
              "- 기존 reports와 원본 logs는 해당 측정 당시의 기록이다. 38 mm를 기준으로 했던 과거 비교·검증 수치는 새 매핑의 검증값으로 사용하지 않는다.", "",
              "[상세 JSON](check75_01_summary.json), [갱신한 기하 매핑](isaac_mapping_fit.md). 실물 동작 명령 없이 로그만 분석했다."]
    (target.parent/"check75_01.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"representative_width_mm":report["pooled_median_mm"],"grouped":grouped,"report":str(target)}))


if __name__ == "__main__": main()
