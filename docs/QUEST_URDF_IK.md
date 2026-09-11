# URDF 기반 Quest 팔 IK — 1차 구현과 확인 방법

대상: `collect_quest_teleop.sh` / `quest_collector.sh collect`의 S200062·S63·S56.
URDF를 읽는 7관절 FK/Jacobian과 bounded IK를 사용한다. NumPy 기반 CPU 계산이며
IK 자체에는 새 패키지 설치가 필요 없다. 기본 켜짐인 자기충돌 필터는 별도 FCL 설치가 필요하다.
브라우저 미리보기는 이번 변경 대상이 아니다.

**현재 상태:** 모델 계산·단위 테스트 완료, 실제 Isaac Sim/Quest 실행은 미확인.
[속도 우선 자기충돌 필터](QUEST_SELF_COLLISION.md)는 S200062 integrated 손에 구현했다.
연속 경로·환경 충돌 회피나 성공률 보장 기능은 없다. S63/S56는 자기충돌 정책 지원 전까지
명시적인 `--no-self-collision` 없이는 수집기를 시작하지 않는다.

## 1. 사용자의 현재 absolute 명령

Runtime과 웹 서버는 그대로 두고 기존 수집기만 종료한 뒤 재실행한다.
웹 클라이언트 재빌드는 필요 없다.

```bash
./quest_collector.sh collect \
  --robot-model s200062 \
  --input-mode controllers \
  --controller-mapping absolute \
  --absolute-orientation downward \
  --arm-response responsive \
  --arm-ik urdf \
  --arm-start-pose ready
```

`--arm-ik auto --arm-start-pose auto`가 기본값이므로 기존 사용자 명령에도 적용된다.
단, **`--scene-config`가 있으면 auto는 그 장면의 초기 팔 자세를 보존**한다.
사용자가 캡처한 자세를 유지하려면 `--arm-start-pose scene`을 명시한다.
ready는 팔을 몸 앞에 굽힌 자세로 생성하므로 주변 rack/box와 간섭하지 않는지 먼저 확인한다.

기존 제어와 비교할 때만 `--arm-ik legacy --arm-start-pose scene`을 사용한다.
불일치 검사를 우회하기 위한 용도로 legacy를 사용하지 않는다.

## 2. 처리 방식

1. 선택된 로봇 URDF에서 어깨 부모 링크→손끝 체인을 읽는다. 7개 관절의 축·원점·한계와
   손끝 고정 변환을 사용한다. 특히 S63 손끝 고정 회전은 다른 모델과 다르다.
2. ready를 선택하면 URDF FK와 bounded 위치 IK로 팔꿈치를 굽힌 준비 자세를 계산한다.
   목표는 어깨 기준 약 앞으로 32cm, 바깥쪽 4cm, 아래로 32cm다. 손목 방향까지 고정한 준비 자세는 아니다.
3. 시작 시 **실제 USD**의 손끝 위치·회전·Jacobian·관절 범위를 URDF와 비교한다.
   검사 실패 시 제어를 시작하지 않고 종료한다. 단일 시작 자세에서의 검사이며
   전체 움직임 범위나 동역학까지 동등함을 보장하지 않는다.
4. 매 제어 tick에서 현재 관절 상태를 읽고 손 목표를 실제 어깨 부모 좌표계로 변환한다.
   관절이 포화되면 해당 관절을 고정하고 나머지 관절로 오차를 다시 푸는 active-set 방식이다.
5. URDF 관절 한계와 live USD 한계의 교집합, 속도·가속도 제한, 관절 정지점 근처 감속을 적용한다.
   정지점 안전과 가속도 제한이 충돌하면 정지점 안전이 우선이다.
6. 초기 관절 자세를 선호하는 항으로 불필요한 관절 회전을 줄이고, 위치 오차가 클수록
   손목 회전 가중치를 줄인다. 위치를 3배 가중하는 **soft priority**이며 엄밀한 계층형 우선순위는 아니다.
7. 관절 목표를 Isaac Lab의 기존 물리 drive에 전달한다. 토크 상한과 중력 보상은 유지한다.

## 3. 도달 범위와 방향의 한계

어깨 중심 반경은 URDF 팔 링크 길이 합의 95%로 계산한다. 이 구 밖의 손 목표는 구 표면으로
투영하고 `[IK] target projection`으로 이동시킨 거리를 보고한다.
이는 **명백히 너무 먼 목표를 거르는 상한**이다. 구 안의 모든 위치·회전 조합이 도달 가능하다는 뜻은 아니다.
관절 범위와 wrist 방향 때문에 목표 오차가 남을 수 있다.

`absolute + downward`는 원하는 손 위치와 아래 방향을 동시에 요청한다. 물리적으로 불가능하면
위치를 우선하고 방향 오차를 허용한다. 아래 방향을 맞추려고 무리하게 팔을 비트는 동작을 줄이는
목적이며, 모든 관절 꼬임을 방지하는 보장은 아니다.
`scaled`는 시작 시 위치·방향 기준을 잡는 매핑을 그대로 사용한다.

몸통·반대팔을 포함한 현재/목표 자세 거리는 별도의 경량 자기충돌 필터로 검사한다.
작업물 장애물 경로 계획과 어깨/팔꿈치의 명시적 해 선택은 구현하지 않았다.
자세 선호 항 자체가 자기충돌 회피를 대신하지는 않는다.

## 4. 사용자가 확인할 순서

1. 시작 로그에서 양팔 모두 `[URDF IK] ... live USD FK/Jacobian/limits matched`인지 확인한다.
   `mismatch` 또는 joint limit 오류가 나오면 그 줄과 Traceback을 보낸다. 임계값을 임의로 늘리지 않는다.
2. 녹화하지 않은 상태에서 양팔 준비 자세와 주변 물체 간섭을 확인한다.
3. 정면을 보고 X로 시점을 맞춘 뒤, 손을 몸 앞에 편하게 두고 A로 따라오기를 켠다.
   absolute는 scaled처럼 시작 위치를 다시 잡지 않으므로 현재 controller 위치로 이동할 수 있다.
4. 컨트롤러 회전을 유지하고 한쪽씩 앞으로 5~10cm 이동한다. 다음으로 작은 회전을 시도한다.
5. 오른팔이 꼬이거나 움직임이 이상하면 A로 정지하고 로그를 보낸다.
   확인 전에는 녹화를 시작하거나 큰 동작을 반복하지 않는다.

| 로그 | 의미 |
| --- | --- |
| `target projection` | URDF 어깨 구 바깥 목표를 줄인 거리. 0이 아니면 목표를 그대로 따르지 않음 |
| `effective position error` | 필터·도달 범위 투영 후 목표와 현재 FK 손끝 사이 오차 |
| `joint limit margin` | 가장 가까운 hard joint limit까지 남은 각도 |
| `rotation weight` | 그 tick에서 적용한 회전 가중치. 멀리 있는 목표는 낮아짐 |
| `parent-frame requested xyz / actual xyz` | 어깨 부모 링크 좌표계에서 필터된 요청 위치와 현재 URDF FK 위치(m) |
| `[MOTION] target error / rotation error` | 원래 매핑 목표와 실제 USD 손끝 간 위치·회전 오차 |
| `[PERF] loop` | 실제 실행 주파수. CPU IK·물리·렌더·스트리밍 전체 부하 영향 |

현재 수집 명령은 scaled가 아니라 **absolute**다. 이전에 설명한 scaled의 시작 위치 문제를
현재 현상의 확정 원인으로 취급하지 않는다. 위 요청/실제 좌표와 관절 여유도를 보고 구분한다.

## 5. 구현 위치와 기록

- `teleop/urdf_arm_ik.py`: URDF FK/Jacobian, 준비 자세, active-set IK와 도달 범위 검사
- `teleop/teleop_ik.py`: live USD 모델 검증, 좌표 변환, 관절 drive 연결
- `teleop/collect_quest_teleop.py`: CLI, 모델 선택, 시작 자세, 진단 로그

파일들은 `src/kuavo_isaaclab_scene/` 아래에 있다. Episode metadata에는 `arm_ik`, `arm_urdf`,
`arm_start_pose`, `arm_reference_joint_positions`를 기록해 기존 제어 데이터와 구분한다.
위치·방향·응답 프로필 선택은 [팔 제어 옵션](QUEST_ARM_CONTROL.md)을 참고한다.
