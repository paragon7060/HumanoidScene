# S63 Leju two-finger 개폐 제어

`--robot-model s63 --gripper leju-twofinger`에서 사용하는 관절 목표 설정이다.
설정의 기준은 `src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json`
한 파일이다. `configs/grippers.json`은 `leju-twofinger` 이름을 이 패키지 preset에
연결만 하므로 같은 값을 두 파일에 중복해서 넣지 않는다.
실물 명령 기준 0은 완전 열림, 100은 완전 닫힘이다. 시뮬레이션 continuous action은
기존과 같이 +1이 열림, -1이 닫힘이며 명령 백분율은 `(1-action)*50`이다.

## 개폐 매핑

양손 모두 끝단 기준 최대 벌림을 약 90 mm로 맞췄다. 완전 열림 관절 목표는
앞 driver -0.363811 rad, 뒤 driver +0.363811 rad다. 다른 gripper preset의 목표는 유지한다.

| 명령 | 닫는 방향의 목표 간격 | 여는 방향의 목표 간격 |
|---:|---:|---:|
| 0 | 90 mm | 90 mm |
| 25 | 78 mm | 68 mm |
| 50 | 51 mm | 40 mm |
| 75 | 26 mm | 18 mm |
| 100 | 약 0 mm | 약 0 mm |

왼손 실측에서 얻은 주 개폐 곡선을 오른손에도 동일하게 적용했다.
표의 모든 오른손 지점을 독립적으로 실측한 것은 아니다.
`sides.left/right.command_scale`로 열린 관절 범위를 확장하고,
`position_mapping`의 닫기·열기 lookup으로 방향별 목표를 계산한다.
중간에 방향을 반전하면 반대쪽 곡선이 이전 목표에 도달할 때까지 목표를 유지해
요청과 반대 방향으로 튀는 것을 방지한다.

## 초기 가속

`sides.left/right.target_filter`는 매핑 이후 관절 목표에 적용하는 가벼운 2단 필터다.
계수는 physics dt가 바뀔 때만 계산하고, 두 단계의 held-target 해를 batched 텐서로 갱신한다.
환경별 Python 반복문이나 영상 처리를 physics loop에 추가하지 않는다.

| 손 | 닫기 시간 상수 / 단계 | 열기 시간 상수 / 단계 | 단계 수 |
|---|---:|---:|---:|
| 왼손 | 0.092 s | 0.076 s | 2 |
| 오른손 | 0.077 s | 0.064 s | 2 |

binary, continuous, RL action에 같은 필터를 적용한다. RL의 기존 증분 입력은 유지한다.
방향 반전 시 중간 필터 목표를 현재값과 새 목표 사이로 제한한다.
reset은 필터 상태를 실제 joint 위치로 초기화하며 RL 중립 입력이 이전 명령을 이어가지 않게 한다.

PD 강성 4000, 감쇠 400과 기존 마찰·구동력 제한은 유지한다.
이 설정은 무부하 개폐의 실용적인 근사이며 물체 접촉력이나 실물 파지력을 보정한 값은 아니다.
검증한 단일 목표 이동에서 기존 대비 encoder 곡선 오차가 61–72% 줄었으며
10–90% 이동 시간은 실물보다 약 6–8% 길다.
GPU 필터 단독 시험의 추가 비용은 양손 합계 약 0.045 ms/physics step였다.
작은 고정 베이스 양손 장면에서 속도 저하는 관찰하지 않았지만 전체 RL throughput 측정은 아니다.

## 실행과 테스트

```bash
./run_scene.sh --robot-model s63 --gripper leju-twofinger
python -m pytest tests/test_gripper_mapping_and_filter.py tests/test_gripper_config.py \
  tests/test_gripper_runtime.py tests/test_gripper_io.py tests/test_twofinger_linkage.py -q
```

테스트는 측정 도구나 측정 데이터 없이 physics dt 독립성, 방향 반전, 환경별 reset,
설정 검증과 확장된 four-bar 구동 범위의 링크 폐합을 확인한다.
