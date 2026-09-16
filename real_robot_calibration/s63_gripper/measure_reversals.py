#!/usr/bin/env python3
"""Operator-paced minor-loop measurement, using the existing single-setpoint tool.

Default: show the plan only. --send requires a recorder-created session folder.
Each movement requires Enter; enter left/right tip widths after settling.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

CASES = (("A_closing_then_opening", [0, 50, 45, 40, 25, 0]),
         ("B_opening_then_closing", [100, 50, 55, 60, 75, 100]))
CHECK75_CASES = (("C_full_closing_75_check", [0, 25, 50, 75, 100]),
                 ("D_reversal_closing_75_check", [100, 50, 55, 60, 75, 100]))


def parse_widths(text):
    values = text.split()
    if len(values) != 2:
        raise ValueError("왼손 오른손 순서로 두 값을 입력하세요. 미측정은 ?")
    result = []
    for value in values:
        if value == "?":
            result.append(None)
            continue
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 200:
            raise ValueError("폭은 0~200 mm 숫자 또는 ?로 입력하세요")
        result.append(number)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Existing output folder of record_real_gripper.py")
    parser.add_argument("--send", action="store_true", help="Enter-confirmed real movements, both hands")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--protocol", choices=("reversal", "check75"), default="reversal")
    parser.add_argument("--measure-only-75", action="store_true", help="check75 only: retain all commands, ask widths only at 75")
    args = parser.parse_args(argv)
    if args.measure_only_75 and args.protocol != "check75":
        parser.error("--measure-only-75 requires --protocol check75")
    cases = CHECK75_CASES if args.protocol == "check75" else CASES
    for name, values in cases:
        print(name + ": " + " -> ".join(map(str, values)))
    print("양손 동일 명령 / velocity 25 / effort 1 A / 각 단계 최소 3초 정지")
    if args.measure_only_75:
        print("폭 입력은 두 경로의 75에서만 받습니다. 나머지는 경로 설정 단계입니다.")
    if not args.send:
        print("계획 미리보기입니다. 실제 측정은 --session 기록폴더 --send")
        return 0
    if args.session is None or not (args.session / "session.json").is_file() or not (args.session / "messages.jsonl").is_file():
        parser.error("먼저 기록기를 실행하고 그 출력 폴더를 --session으로 지정하세요")
    output = args.session / "reversal_measurements.jsonl"
    # A new invocation gets a new session; never silently overwrite or append trials.
    if output.exists():
        parser.error("이 세션에 reversal 기록이 이미 있습니다. 새 기록 세션을 사용하세요")
    controller = Path(__file__).resolve().parent / "control_real_gripper.py"
    session_info = json.loads((args.session / "session.json").read_text())
    host_options = ["--host", session_info["host"], "--workspace", session_info["workspace"],
                    "--ros-master", session_info["ros_master"]]
    with output.open("x") as log:
        def emit(kind, **fields):
            log.write(json.dumps({"kind": kind, "local_utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")
            log.flush()
        emit("protocol", protocol=args.protocol, cases=[{"case": name, "commands": values} for name, values in cases],
             repeats=args.repeats, sides=["left", "right"], velocity=25, effort_A=1,
             measure_only_percent=75 if args.measure_only_75 else None,
             tip_reference="Same distal tip surfaces as previous width measurement")
        try:
            for trial in range(1, args.repeats + 1):
                for case, values in cases:
                    for step, percent in enumerate(values, 1):
                        context = {"trial": trial, "case": case, "step": step, "command_percent": percent}
                        reply = input(f"\n[{trial}/{args.repeats} {case} {step}/{len(values)}] 양손 {percent} 이동: Enter / 종료 q: ")
                        if reply.strip().lower() == "q":
                            emit("operator_stopped", **context)
                            return 0
                        emit("step_attempt", **context)
                        command = [sys.executable, str(controller), "--position", str(percent),
                                   "--velocity", "25", "--effort", "1", "--send", "--log",
                                   str(args.session / "commands.jsonl"), *host_options]
                        result = subprocess.run(command)
                        if result.returncode:
                            emit("command_failed_or_uncertain", returncode=result.returncode, **context)
                            print("명령 실패 또는 결과 불확실: 측정을 중단합니다. 자동 재전송은 없습니다.", file=sys.stderr)
                            return 1
                        print("3초 정지 후 폭을 측정하세요. 계속 움직이면 충분히 멈출 때까지 기다리세요.")
                        time.sleep(3)
                        if args.measure_only_75 and percent != 75:
                            emit("width_not_requested", **context)
                            continue
                        while True:
                            try:
                                widths = parse_widths(input("끝단 폭 mm — 왼손 오른손 (예: 51 49 / 미측정 ?): "))
                                break
                            except ValueError as error:
                                print(error)
                        note = input("비고(없으면 Enter): ")
                        emit("width_measurement", left_width_mm=widths[0], right_width_mm=widths[1],
                             note=note, width_time_source="Local input receipt, not exact physical measurement instant", **context)
            emit("completed")
        except (KeyboardInterrupt, EOFError):
            emit("interrupted")
            print("저장된 측정은 유지됩니다. 이 종료는 이미 수락된 로봇 동작을 취소하지 않습니다.", file=sys.stderr)
            return 1
    print(f"측정 저장 완료: {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
