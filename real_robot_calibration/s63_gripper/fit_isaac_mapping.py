#!/usr/bin/env python3
"""Reproduce the left S63 command fit from measurements and packaged CAD.

Run with the Isaac Lab Python (NumPy required). No simulator or robot access.
Writes the two gripper configuration copies and an auditable comparison report.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kuavo_isaaclab_scene.robots.claw_assets.geometry import TwoFingerGeometry


def fit(measurements):
    geometry = TwoFingerGeometry("left")
    samples = measurements["directional_width_samples"]
    opened = geometry.driver_for_gap_m(measurements["tip_opening"]["reported_width_mm"] / 1000)
    percent = np.arange(101, dtype=float)
    mapping = {"command_percent": percent.tolist()}
    rows = []
    for direction in ("closing", "opening"):
        sample = samples[direction]
        order = np.argsort(sample["command_percent"])
        x = np.asarray(sample["command_percent"])[order]
        widths = np.asarray(sample["width_mm"])[order]
        angles = np.array([geometry.driver_for_gap_m(w / 1000) for w in np.interp(percent, x, widths)])
        mapping[f"{direction}_fractions"] = ((angles - opened) / -opened).tolist()
        for p, width in zip(sample["command_percent"], sample["width_mm"]):
            angle = float(angles[int(p)])
            rows.append({"direction": direction, "command_percent": p,
                         "measured_width_mm": width,
                         "original_isaac_width_mm": geometry.gap_m(-.25 * (1 - p / 100)) * 1000,
                         "calibrated_driver_front_rad": angle,
                         "calibrated_driver_back_rad": -angle,
                         "calibrated_geometric_width_mm": geometry.gap_m(angle) * 1000})
    return {"command_scale": opened / -.25, "position_mapping": mapping}, {
        "robot_model": "s63", "preset": "leju-twofinger", "side": "left",
        "geometry_sha256": geometry.sha256,
        "tip_reference": "Inner mesh edges in distal local-Z 1 mm band, projected on base X",
        "physical_tip_surface_confirmed": False,
        "open_driver_front_rad": opened, "open_driver_back_rad": -opened,
        "fit": "Piecewise-linear measured width versus command; inverse four-bar CAD sampled every 1 percent",
        "reversal_rule": "Hold previous joint target until opposite direction envelope reaches it; closing-75 corrected to repeated 26 mm reference",
        "closing_75_source": measurements.get("closing_75_correction"),
        "limitations": ["Closing-75 repeated three times per approach on both hands; other full-cycle nodes and reversal anchors not repeatedly measured",
                        "Static gap fit only; velocity, delay, gains, torque and contact unchanged",
                        "Right hand adopts left mapping except closing-75 is now independently supported; other right full-cycle nodes not independently measured",
                        "CAD closed tip residual is about 0.025 mm; user reports rounded zero"],
        "comparison": rows}


def main():
    folder = Path(__file__).resolve().parent
    measurements = json.loads((folder / "measured_parameters.json").read_text())
    if measurements["tip_opening"]["side"] != "left":
        raise ValueError("This fit is explicitly scoped to the measured left hand")
    calibration, report = fit(measurements)
    applied_sides = ["left"]
    if measurements.get("mapping_application", {}).get("right") == "copy_left_by_user_request":
        applied_sides.append("right")
    report["applied_sides"] = applied_sides
    report["right_hand_full_cycle_independently_measured"] = False
    for path in (ROOT / "configs/grippers.json", ROOT / "src/kuavo_isaaclab_scene/configs/grippers.json"):
        source = path.read_text()
        for side in applied_sides:
            start = source.index("{", source.index(f'"{side}":', source.index('"leju-twofinger":')))
            hand, length = json.JSONDecoder().raw_decode(source[start:])
            hand.update(calibration)
            replacement = json.dumps(hand, indent=2).replace("\n", "\n        ")
            source = source[:start] + replacement + source[start + length:]
        path.write_text(source)
    reports = folder / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "isaac_mapping_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# S63 왼손 실측–Isaac 매핑 보정", "", f"열림 driver: front={report['open_driver_front_rad']:.8f} rad, back={report['open_driver_back_rad']:.8f} rad.", "",
             "| 방향 | 명령 % | 실측 mm | 기존 Isaac 기하 mm | 보정 기하 mm | front rad |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["comparison"]:
        lines.append(f"| {row['direction']} | {row['command_percent']} | {row['measured_width_mm']} | {row['original_isaac_width_mm']:.3f} | {row['calibrated_geometric_width_mm']:.3f} | {row['calibrated_driver_front_rad']:.6f} |")
    lines += ["", "## 적용", "", f"configs/grippers.json 및 패키지 사본의 leju-twofinger에 적용한 손: {', '.join(applied_sides)}. 오른손은 사용자 요청으로 왼손 매핑을 복사했으며, 닫기 75만 양손 독립 반복 측정으로 뒷받침된다. 다른 오른손 전체 왕복 노드는 독립 실측이 아니다. signed action +1=명령 0(열림), -1=명령 100(닫힘). 실측 폭을 구간 선형 보간한 뒤 four-bar 기하의 역변환으로 관절 목표를 산출한다. 1% 간격 lookup을 runtime에서 보간하므로 사이 구간에는 작은 근사 오차가 있다.", "",
              "같은 명령을 유지하면 목표도 유지한다. 중간 방향 전환에서는 반대 곡선이 현재 목표에 도달할 때까지 유지하는 기존 규칙을 사용한다. 작은 역방향 명령의 유지 구간은 실측 오차와 하드웨어 피드백으로 뒷받침된다. 닫기 75는 전체 닫기/역전 경로를 양손 각각 3회 재측정한 대표값 26 mm로 보정했다. 기존 38 mm는 원본 관측으로 보존하되 현재 fitting에는 사용하지 않는다. 별도의 역전 곡선은 적용하지 않는다.", "",
              "## 측정 기준과 한계", "", report["tip_reference"] + ". 실물의 안쪽/바깥쪽 측정 면은 아직 명시적으로 확인되지 않았다.", "",
              *["- " + limitation for limitation in report["limitations"]], "",
              "동적 Isaac 검증 결과는 isaac_mapping_validation.json에 별도로 기록한다."]
    (reports / "isaac_mapping_fit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"open_rad": report["open_driver_front_rad"], "command_scale": calibration["command_scale"], "report": str(reports / "isaac_mapping_fit.md")}))


if __name__ == "__main__":
    main()
