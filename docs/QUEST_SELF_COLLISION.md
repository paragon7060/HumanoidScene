# Quest 자기충돌 검사 — 속도 우선 경량 모드

`quest_collector.sh collect`의 **S200062 + integrated gripper** 수집기에 기본 적용된다.
브라우저 미리보기에는 적용하지 않는다. 다른 모델·외장 손은 현재 형상 정책의 지원 대상이
아니므로 시작 시 오류를 낸다. 검사 없이 실행하려면 명시적으로 `--no-self-collision`을 사용한다.

## 실행

새 PC에서 최초 한 번:

```bash
./setup_self_collision.sh
```

Isaac Lab Python을 찾아 `python-fcl==0.7.0.8`만 `.external/self-collision`에 설치한다.
공유 NumPy/Torch를 업그레이드하지 않는다. NumPy/SciPy/trimesh는 기존 환경에 있어야 한다.
현재 작업 PC에는 설치했다. 형상 거리 질의는 [python-fcl](https://github.com/BerkeleyAutomation/python-fcl)을 사용한다.

Runtime과 웹 서버를 준비한 뒤 기존 실행 명령을 그대로 사용한다:

```bash
./quest_collector.sh collect \
  --input-mode controllers \
  --controller-mapping absolute \
  --absolute-orientation downward \
  --arm-response responsive
```

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--self-collision` | 켜짐 | 제어 tick마다 현재·목표 자세를 검사하는 경량 필터 |
| `--no-self-collision` | 꺼짐 | 검사 완전 해제. 성능 비교용이며 충돌 제한도 없어짐 |
| `--self-collision-clearance` | `0.003` m | 허용 접촉을 제외한 모델 형상 간 최소 간격. 너무 크게 하면 초기 자세도 거부될 수 있음 |

`--arm-ik legacy`는 충돌 검사를 끄지 않는다. IK와 충돌 필터는 별개다.

## 비용을 제한하는 방법

- **제어 명령이 갱신될 때만** 검사한다. 기본 30 Hz 제어 / 120 Hz 물리라면 4개 물리 스텝 중 첫 스텝에서만 계산한다.
- 현재 자세와 다음 관절 목표, 최대 **두 번의 거리 질의**만 한다. 정지 목표라면 한 번이다.
- 전체 모델 쌍은 저렴한 bounding-sphere 거리로 먼저 거르고, 가까운 후보만 FCL로 검사한다.
- 런타임에서는 충돌 Jacobian, 충돌 회피 최적화, 반복 경로 샘플링, 관성 예측을 실행하지 않는다.
  소스에 남은 엄격한 `filter`/swept-path 함수는 오프라인 비교용이며 수집기는 `filter_light`만 호출한다.
- 중간 물리 스텝은 이전 보정 명령을 재사용한다. 베이스 이동과 바퀴 속도 명령은 유지한다.

목표가 너무 가까우면 그 tick의 관절 위치 명령을 현재 자세로 유지한다. 우회 경로를 찾거나
오차를 장시간에 걸쳐 줄이는 충돌 최적화는 없다. 기존 IK의 응답 설정은 별도로 유지되며,
필터는 URDF 관절 범위와 최대 `min(URDF 속도, 2.5) × 제어 dt` 이동량을 적용한다.

**한계:** 두 끝 자세 사이의 충돌, 중간 물리 스텝에서 발생한 충돌, 관성·외력에 의한 침범은
놓칠 수 있다. 현재 자세의 간격 위반을 다음 제어 tick에서 감지하면
`[SELF COLLISION STOP]` 오류로 수집기를 중단한다. 실제 로봇 안전 기능이 아니며 무충돌 보장이 아니다.

## 형상 범위

현재 번들 S200062 모델은 58개 형상, 허용 쌍 제외 1,454개 후보 쌍을 다룬다.
양팔·반대 손·머리·몸통·하체·베이스를 포함한다. 같은 강체/인접 관절 링크는 제외하고,
four-bar 조립부와 같은 손의 닫힘 접촉 등은
`src/kuavo_isaaclab_scene/configs/self_collision_s200062.json`에 이유와 함께 명시했다.
이 허용은 해당 링크 쌍 전체에 적용되며 임의 초기 자세를 보고 자동으로 예외를 추가하지 않는다.

URDF collision 형상을 우선하고 없으면 visual 형상의 **볼록 껍질**을 사용한다.
이는 모든 실제 USD 메시의 정밀 충돌 검사와 동일하지 않다. 형상이 없는 참조 링크는
시작 로그에 표시한다. 랙·상자·운반 중인 물체·사람 등 **외부 환경 충돌은 검사하지 않는다**.

시작 시 USD의 joint/body 이름과 URDF의 FK가 맞는지 검사한다.
`mismatch`, 누락 메시, 초기 자세 충돌 오류가 나오면 해당 로그를 보내고 정지 상태에서
원인을 확인한다. 이 검사도 실제 USD의 모든 메시 형상이 같다는 증명은 아니다.

## 기록과 확인

기존 `action`은 사용자 입력 의미를 유지한다. 충돌 필터 출력은 별도로 기록한다:

| LeRobot feature | 내용 |
| --- | --- |
| `observation.self_collision.safe_joint_target` | 필터가 출력한 전체 관절 위치 목표. 이름 순서는 feature metadata에 기록 |
| `observation.self_collision.modified` | 해당 제어 tick에서 명령이 변경됐는지 |
| `observation.self_collision.minimum_distance` | 검사한 현재 자세의 최소 간격(m). 먼 쌍은 보수적인 구 간격일 수 있음 |

`safe_joint_target`이라는 이름은 연속 경로 안전 보장을 뜻하지 않는다.
HDF5에도 대응하는 `self_collision_*` 필드를 저장하고 session metadata에는
`self_collision_mode=light_control_endpoints`, 관절 이름, 허용 쌍, clearance를 기록한다.
**기존 feature schema의 LeRobot 데이터셋에 이어 쓰지 말고 새 dataset 경로를 사용한다.**

Isaac Sim/Quest의 실제 FPS·지연 및 시각 검증은 수행하지 않았다. CPU 단독 테스트 시간과
렌더링·GPU 동기화를 포함한 `[PERF] loop`는 다르다. 사용자가 기존 명령으로 실행하여
녹화 전 작은 팔 동작과 `[PERF] loop`, `[SELF COLLISION]` 로그를 확인한다.

2026-09-07 현재 PC, 번들 S200062 준비 자세, 10회 warm-up 후 100회 CPU 단독 측정:
정지 목표 평균 **1.61 ms**, 전 관절 `+0.001` 목표 평균 **3.18 ms** / p95 **3.22 ms**.
이는 30 Hz 제어 tick당 비용이며 120 Hz 물리 스텝마다 발생하는 비용이 아니다.
한 자세의 측정으로 최악 시간이나 FPS 유지까지 보장하지 않는다. 실제 실행에는 GPU→CPU
관절 읽기와 drive 쓰기 비용도 추가된다.
