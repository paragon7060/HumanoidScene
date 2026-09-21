# VR 관절 응답 기록과 real/sim 비교

**실물 `kuavo_hand_track_MR` 앱으로 이미 로봇을 조작하는 경우에는 [실물 VR 기록 절차](REAL_VR_RESPONSE_CALIBRATION.md)를 우선한다.** 이 문서는 Isaac 전용 VR 기록과 검토용 후보 생성의 보조 경로다. 실물 VR 앱의 입력/상태 기록이나 실물 로그의 sim 재생을 구현한 도구가 아니다.

프로젝트 VR teleop은 Isaac에서만 동작한다. VR로 편하게 움직이면서 관절 목표와 sim의 실제 q/v를 기록하고, 그중 작은 움직임을 실물 비교 후보로 추출한다. 컨트롤러를 사람 손으로 완전히 정지시킬 필요는 없다. Gripper 보정은 별도 작업에서 진행한다.

현재 구현 범위는 **VR 기록 → 구간 검사 → 실물 비교 후보 생성 → 동일 후보의 Isaac 재생**이다. 후보는 검토용이며 `calibrate.py run --send`에서 거부된다. 실물 송신 기능은 이 후보에 연결하지 않았다. 실제 공간·기준 자세의 충돌/균형 검토, 최신 실물 캡처와 입력 소유권 확인 후에 실물 재생 경로를 연결한다.

## 1. 먼저 VR 기록

```bash
cd /path/to/HumanoidScene
VR_LOG="$PWD/recordings/joint_response_$(date +%Y%m%d_%H%M%S).jsonl"
echo "$VR_LOG"
./quest_collector.sh collect \
  --robot-model s63 --gripper leju-twofinger --device cuda:0 \
  --input-mode controllers --controller-mapping absolute \
  --absolute-orientation downward --arm-response responsive \
  --no-rack-rollers --rl-reward-debug 1 \
  --no-quest-camera-overlay --no-camera-preview \
  --joint-response-log "$VR_LOG"
```

기존 Quest 연결을 사용한다. **A/T로 시작**, 팔을 천천히 움직이고 **A/T로 pause**, 프로그램은 기존 종료 방법 또는 Ctrl+C로 정상 종료한다. 처음에는 10~20초씩 2~3구간이면 충분하다. 바퀴/base stick 입력은 주지 않고 물체·랙에 닿지 않는 공간에서 움직인다. 관절 오차가 컸던 왼팔 5번과 오른팔 5·6·7번의 변화가 들어가면 우선 검토하기 좋다. 이번 기록은 Isaac만 움직인다.

일반 수집에서도 동일한 `--joint-response-log` 옵션을 쓴다. 위 명령의 `--rl-reward-debug 1`을 빼면 일반 수집이며 **T로 motion preview**를 켜면 이미지 데이터셋 없이 응답을 기록할 수 있다. RL debug에서는 이미지 데이터셋을 만들지 않아도 응답 JSONL은 기록된다. 파일이 이미 존재하면 보존하고 새 이름을 요구한다.

몸통·양팔 관절 기록은 모든 모델에 적용된다. 최초 실물 후보 생성과 후보 sim 재생은 S63 팔 한 관절에 한정한다. 다른 모델의 실측값을 S63에서 가져오지 않는다.

## 2. 기록 내용과 구간 검사

```bash
python3 real_robot_calibration/s63_body_arms/vr_trajectory.py inspect \
  --log "$VR_LOG" > "${VR_LOG%.jsonl}_inspect.json"
```

JSONL에는 관절 이름/순서, 한 제어 단계 전후의 q/v, 논리 q/v 목표, 마지막 physics write의 solver 목표·중력 bias, root pose, sim/벽시계 시각, 실제 sim gain과 추정 토크가 들어간다. 값은 rad/rad/s이고 검사 보고서의 오차는 degree다. 재시작/reset/recenter/추적 손실·중단은 구간을 나눈다. `segments`의 `exportable_steps`를 보고 연속 유효 구간을 고른다. reset·추적 손실·충돌이 섞이면 `--start-s/--end-s`로 유효 범위만 지정한다.

**논리 목표가 실물 명령의 원본이다.** 중력 보상을 위한 solver 목표/bias를 실물에 보내면 실물 WBC 보상과 중복되므로 후보에서 제외한다. 토크는 Isaac의 추정값이고 실물 출력 토크 측정이 아니다. 한 제어 단계의 마지막 목표만 기록하므로 IK가 physics substep마다 갱신한 전체 명령열을 복원하지 않는다. 검사 오차는 마지막 목표 대비 단계 끝 q이며 순수 모터 지연이나 실제 Kp/Kd 식별 결과가 아니다.

## 3. 실물 비교 후보 생성 — 아직 송신하지 않음

VR 로그 검토 후 실물의 정상 quick 목표를 읽기 전용으로 캡처한다. 아래 `--segment 3`은 예시이며 검사 결과의 실제 번호로 바꾼다.

```bash
CAL=real_robot_calibration/s63_body_arms/calibrate.py
VR_TOOL=real_robot_calibration/s63_body_arms/vr_trajectory.py
SESSION="$PWD/real_robot_calibration/s63_body_arms/data/vr_$(date +%Y%m%d_%H%M%S)"
python3 "$CAL" capture --output "$SESSION/ready.json"
python3 "$VR_TOOL" export --log "$VR_LOG" --capture "$SESSION/ready.json" \
  --segment 3 --joint zarm_l5_joint --scale 1 --time-scale 5 \
  --waypoint-s 1 --output "$SESSION/l5_candidate.json"
```

팔 하나의 **한 관절만** 첫 목표 대비 offset을 추출해 실물 baseline에 더한다. source 전체에서 ±1°를 넘으면 거부한다. 필요하면 짧은 범위를 선택하거나 `--scale 0.1`처럼 축소를 명시한다. 다른 팔/몸통의 원본 움직임은 후보에 보내지 않으며 제외한 변화량은 metadata에 남긴다.

1초 간격 waypoint를 quintic으로 연결하고 시간을 늘린다. 각 ramp는 최소 2초, 목표 최대 속도 0.469°/s, 가속도 0.361°/s² 이하, 기준 대비 ±1° 이하이며 URDF 한계에 0.5° 여유를 요구한다. 초기 hold 3초 → 추출 궤적 → baseline 복귀 최소 4초 → 최종 hold 3초다. JSON과 50Hz q/v CSV를 새 파일로 만든다. 이 제한은 현장 충돌/균형 안전성을 입증하지 않는다.

## 4. 가공한 동일 후보를 Isaac에서 재생

```bash
bash scripts/replay_joint_candidate.sh \
  --robot-model s63 --gripper leju-twofinger --device cuda:0 \
  --candidate "$SESSION/l5_candidate.json" \
  --output "$SESSION/l5_processed_sim.jsonl"
python3 "$VR_TOOL" inspect --log "$SESSION/l5_processed_sim.jsonl" \
  > "$SESSION/l5_processed_sim_inspect.json"
```

headless Isaac에서 동일한 절대 q/v를 현재 gain·중력 보상 설정으로 재생한다. baseline 몸통·양팔은 후보 값으로 초기화한다. ROS/SSH/실물 명령을 사용하지 않는다. 원본 로그에서 후보를 재구성해 편집된 waypoint나 바뀐 source를 거부한다. `--max-steps 10`은 시작 확인용이며 보고서 `complete=false`가 된다. 완주 검증을 대신하지 않는다.

완료 여부와 예외는 응답 파일 옆의 `.summary.json`에 저장한다. 응답 JSONL의 `closed=true`는 파일이 닫혔다는 뜻이고 궤적 완주를 뜻하지 않는다. 반드시 summary의 `complete=true`와 `error` 없음까지 확인한다. Head와 gripper는 sim 초기 목표를 유지한다.

2026-09-16 합성 20초 후보로 S63 + leju-twofinger의 실제 Isaac 재생을 검증했다. CUDA PhysX는 601단계를 완주했고 논리 목표 변화는 0이었다. CPU PhysX에서는 약 3.7초에 `wheel_left_behind_joint` q/v가 NaN이 되어 중력 보상 검사가 중단됐다. 바퀴 문제를 무시하거나 NaN을 0으로 치환하지 않았다. 따라서 이번 절차는 CUDA를 기준으로 안내하며 CPU 재생의 해당 문제는 미해결이다. CUDA 후보 재생 성공은 XR 연결/VR 실제 조작이나 실물 검증 결과가 아니다. [검증 기록](../real_robot_calibration/s63_body_arms/reports/vr_pipeline_20260916.md)을 참고한다.

실물 비교에는 **원래 VR 속도의 sim 기록이 아니라 이 가공 후보의 sim 기록**을 사용해야 한다. 한 관절 추출·시간 변경·quintic 보간 때문에 둘은 다른 실험이다. 이 도구는 작업셀 충돌 검사나 실제 섀시 균형을 검증하지 않으며 결과에 이를 명시한다.

## 이후 비교 및 보정

후보의 baseline/공간 검토 후, 기존 실물 송신 경로에 같은 q/v와 중단 감시를 유지한 재생 기능을 연결한다. 현재는 실제 VR 로그 검토를 먼저 한다. 실물은 논리 목표 → quick/WBC 목표 → joint_cmd → 실제 q/v/IMU를 함께 기록해야 필터 지연과 관절 응답을 구분할 수 있다. 원격 receipt와 ROS header를 검토한 뒤 시간 정렬한다.

처음에는 equivalent 추종 오차, 방향 차이, 정착/복귀, 속도별 lag를 비교한다. q_cmd/q만으로 질량·COM·마찰·실제 motor gain을 각각 확정할 수 없다. 중력/하중 보정은 다른 자세·하중 자료와 단위가 검증된 effort가 추가로 필요하다. 이 작은 free-space 시험만으로 conveyor 접촉이나 전체 작업 자세 전환을 검증했다고 해석하지 않는다.
