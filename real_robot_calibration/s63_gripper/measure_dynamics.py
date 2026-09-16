#!/usr/bin/env python3
"""Operator-paced free-space time-response capture; default is preview only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def make_plan(velocities, repeats):
    return [dict(velocity=velocity, trial=trial, step=step,
                 command_percent=percent, measured_transition=step in (3, 4))
            for velocity in velocities for trial in range(1, repeats + 1)
            for step, percent in enumerate((0, 25, 75, 25), 1)]


def recording_issue(messages):
    if not messages.is_file() or time.time() - messages.stat().st_mtime > 30:
        return "상태 파일이 30초 이상 갱신되지 않았습니다. 터미널 1의 기록을 확인하세요"
    with messages.open("rb") as stream:
        stream.seek(max(0, messages.stat().st_size - 65536))
        lines = stream.read().splitlines()[1:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if any(row.get("kind") == "summary" for row in rows):
        return "기록이 이미 종료되었습니다. 새 기록 세션을 실행하세요"
    if not any(row.get("topic") == "/leju_claw_state" for row in rows):
        return "상태 기록을 아직 확인하지 못했습니다. 터미널 1의 연결과 기록을 확인하세요"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--velocities", type=int, nargs="+", choices=(25, 50, 75), default=[25])
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if len(set(args.velocities)) != len(args.velocities):
        parser.error("속도 중복 지정은 안 됩니다")
    plan = make_plan(args.velocities, args.repeats)
    print(f"양손 / effort 1 A / 속도 {args.velocities} / 각각 {args.repeats}회")
    print("매 반복 0→25 (시작 준비) →75→25 (응답 측정), 각 명령 뒤 최소 5초 기록")
    print(f"총 {len(plan)}개 명령, 측정 전환 {sum(r['measured_transition'] for r in plan)}개")
    if not args.send:
        print("미리보기입니다. 움직이지 않습니다. 실제 측정: --session 새 기록폴더 --send")
        return 0
    if args.session is None or not (args.session / "session.json").is_file() or not (args.session / "messages.jsonl").is_file():
        parser.error("먼저 record_real_gripper.py를 실행하고 그 출력 폴더를 지정하세요")
    messages = args.session / "messages.jsonl"
    # Inspect only a small tail; the complete raw recording can be very large.
    issue = recording_issue(messages)
    if issue:
        parser.error(issue)
    output = args.session / "dynamics_measurements.jsonl"
    if output.exists() or (args.session / "commands.jsonl").exists():
        parser.error("기존 명령/측정이 있는 세션입니다. 새 세션을 사용하세요")
    info = json.loads((args.session / "session.json").read_text())
    host_options = ["--host", info["host"], "--workspace", info["workspace"], "--ros-master", info["ros_master"]]
    controller = Path(__file__).resolve().with_name("control_real_gripper.py")
    with output.open("x") as log:
        def emit(kind, **fields):
            log.write(json.dumps(dict(kind=kind, local_utc=datetime.now(timezone.utc).isoformat(), **fields), ensure_ascii=False) + "\n")
            log.flush()
        emit("protocol", protocol="dynamics", velocities=args.velocities, repeats=args.repeats,
             effort_A=1, hold_s=5, sides=["left", "right"], plan=plan)
        try:
            for row in plan:
                label = "응답 측정" if row["measured_transition"] else "시작 준비"
                while True:
                    reply = input(f"\n[속도 {row['velocity']} / {row['trial']}/{args.repeats}회 / {label}] 양손 {row['command_percent']}: Enter 이동 / q 종료: ").strip().lower()
                    if reply in ("", "q"):
                        break
                    print("이동은 Enter, 종료는 q를 입력하세요.")
                if reply == "q":
                    emit("operator_stopped", **row)
                    return 0
                issue = recording_issue(messages)
                if issue:
                    emit("recording_unavailable", reason=issue, **row)
                    print(issue + " — 다음 동작을 보내지 않고 중단합니다.", file=sys.stderr)
                    return 1
                emit("step_attempt", **row)
                result = subprocess.run([sys.executable, str(controller), "--position", str(row["command_percent"]),
                                         "--velocity", str(row["velocity"]), "--effort", "1", "--send",
                                         "--log", str(args.session / "commands.jsonl"), *host_options])
                if result.returncode:
                    emit("command_failed_or_uncertain", returncode=result.returncode, **row)
                    print("명령 실패/결과 불확실: 중단합니다. 자동 재전송은 없습니다.", file=sys.stderr)
                    return 1
                print("5초 기록 중입니다. 다음 Enter는 양손이 멈춘 뒤 누르세요.")
                time.sleep(5)
                emit("hold_completed", **row)
            emit("completed")
        except (KeyboardInterrupt, EOFError):
            emit("interrupted")
            print("기록은 유지됩니다. 종료는 이미 수락된 로봇 움직임을 취소하지 않습니다.", file=sys.stderr)
            return 1
    print(f"측정 완료: {output.resolve()}\n터미널 1 기록기를 Ctrl+C로 종료해도 수집된 로그는 저장됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
