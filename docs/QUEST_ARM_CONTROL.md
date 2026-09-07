# Quest 팔 제어 옵션: 위치·방향·응답 속도

이 문서는 `collect_quest_teleop.sh`와 `quest_collector.sh collect`의 팔 제어 설정이다.
브라우저 미리보기(`preview_quest_browser.sh`)의 팔 매핑에는 적용되지 않는다.
현재 scaled/absolute/맨손은 기본적으로 URDF bounded IK를 사용한다.
새 제어의 확인 순서와 아직 없는 기능은 [URDF IK 가이드](QUEST_URDF_IK.md)를 먼저 참고한다.
연결 준비는 [수집기 설치·실행](QUEST_COLLECTOR_SETUP.md), 버튼 조작은
[Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)를 참고한다.

## 1. 바로 실행하기

Runtime과 웹 서버가 준비된 상태에서, 기존 scene·robot·dataset 옵션을 유지하며 실행한다.

```bash
# scaled: 기준 자세에서 손 이동을 1.1배 반영 + 빠른 팔 응답
./quest_collector.sh collect \
  --input-mode controllers \
  --controller-mapping scaled \
  --position-gain 1.1 \
  --arm-response auto
```

`auto`는 **scaled와 absolute 컨트롤러 모두 responsive**를 선택한다.
따라서 기존 scaled 실행 명령도 수집기를 재시작하면 빠른 응답을 사용한다.
명령에 `--arm-response smooth`를 명시했다면 해당 값이 우선한다.

```bash
# absolute: VR grip 위치 1:1 + 정면 컨트롤러 → 아래 방향 그리퍼
./quest_collector.sh collect \
  --input-mode controllers \
  --controller-mapping absolute \
  --absolute-orientation downward \
  --arm-response auto
```

이전 팔 응답과 비교하려면 `--arm-response smooth`로 바꾼다.
먼저 녹화 없이 A로 따라오기를 켜고, 물체와 떨어진 곳에서 작은 이동·회전을 확인한다.
변경은 시뮬레이션용이며 실제 로봇 하드웨어에 적용할 게인 설정이 아니다.

## 2. 위치 매핑과 팔 응답은 별개

| 모드 | 손 위치를 로봇 목표로 바꾸는 방법 | 회전 | `auto` 팔 응답 |
| --- | --- | --- | --- |
| 컨트롤러 `scaled` (기본) | A 재개 시 사람 손과 로봇 손끝을 기준으로 잡고, 몸통 기준 변위 × `position-gain` | 기준 자세에서 회전 변화 1:1 | `responsive` |
| 컨트롤러 `absolute` | head-camera에 맞춘 XR 가상 월드 grip 위치를 로봇 목표에 1:1 대응 | `absolute-orientation` 선택 | `responsive` |
| 컨트롤러 `relative` | 이전 입력 대비 이동·회전량 누적, 구형 방식 | `rotation-gain` 적용 | `smooth` |
| 맨손 `--input-mode hands` | 별도 scaled wrist 매핑 사용; `controller-mapping`으로 absolute가 되지 않음 | 기준 자세에서 회전 변화 1:1 | `smooth` 유지; 필요 시 `responsive` 명시 |

`position-gain 1.1`은 손을 10cm 움직이면 **목표를 11cm 이동**시키는 설정이다.
`arm-response`는 그 목표를 로봇 팔이 **얼마나 빠르게 따라가는지** 정한다.
gain을 높여 추종 지연을 보정하면 목표 위치와 떨림도 확대되므로 서로 대체하는 옵션이 아니다.

## 3. 옵션 사전

아래 기본값은 Python 수집기 기준이며 스크립트의 차이는 별도로 표시했다.

| 옵션 | 기본값 / 선택값 | 무엇을 바꾸는가·주의점 |
| --- | --- | --- |
| `--input-mode` | `controllers`; `hands` | 컨트롤러 grip+trigger 또는 맨손 wrist+pinch 입력 |
| `--controller-mapping` | `scaled`; `absolute`, `relative` | 컨트롤러의 위치·회전 매핑 방식. 위 표 참고 |
| `--position-gain` | `1.1` | scaled/맨손 목표 변위 배율(1.0~3.0). absolute에서는 무시. relative에도 적용 |
| `--rotation-gain` | `1.0` | 구형 relative 팔 회전 배율만 변경. scaled/absolute/맨손은 회전 1:1 |
| `--absolute-orientation` | `downward`; `pointing` | absolute 컨트롤러만 적용. scaled/맨손의 방향을 바꾸지 않음 |
| `--arm-response` | `auto`; `responsive`, `smooth` | IK 필터·오차 보정 이득·DLS damping·관절 속도/가속도 상한을 묶어서 선택. 명시한 프로필은 입력 모드 전환 후에도 유지 |
| `--arm-ik` | `auto`; `urdf`, `legacy` | auto는 scaled/absolute/맨손에 URDF bounded IK, relative에 기존 IK. URDF 모드는 시작 시 USD 일치 검사 |
| `--arm-start-pose` | `auto`; `ready`, `scene` | auto는 URDF 모드이며 custom scene-config가 없을 때 준비 자세 생성. scene은 장면의 초기 팔 관절 보존 |
| `--self-collision` / `--no-self-collision` | 켜짐 | S200062 integrated 손 전용. 제어 tick마다 가까운 후보의 현재·목표 자세만 검사. 녹화 중 충돌은 episode만 실패 종료하며 Quest/프로그램은 유지. 미녹화 중에는 VR 알림만 표시. [범위·설치·성능](QUEST_SELF_COLLISION.md) |
| `--self-collision-clearance` | `0.003` m | 허용 접촉 제외 모델 형상의 최소 간격. 녹화 중 간격 위반 목표는 해당 tick에서 보류 |
| `--arm-orientation-weight` | `0.5`, 범위 0~1 | IK에서 회전 추종 비중. `0`은 위치만 추종하는 진단 설정. 회전 배율이나 고정 방향 잠금이 아님 |
| `--arm-stiffness` | `800`, 양수 | 물리엔진 팔 관절 drive의 위치 강성. Cartesian IK의 오차 보정 이득과 별개 |
| `--arm-damping` | `50`, 0 이상 | 물리엔진 관절 drive의 감쇠. 아래 DLS damping과 별개 |
| `--control-hz` | Python 직접 실행 `60`; 간편 실행기 `quest_collector.sh collect`는 `30`; 선택 30/60 | 시뮬레이션 제어 주기. 실제 처리 FPS 보장이 아님. 간편 실행기에도 명시적으로 덮어쓸 수 있음 |

`downward`는 컨트롤러를 수평으로 정면에 향할 때 그리퍼가 아래를 향하게 하는
고정 로컬 회전 오프셋이다. 이후 pitch/roll/yaw도 컨트롤러를 따라간다.
항상 월드 아래로 잠그는 기능은 아니다. `pointing`은 이전 정면 접근 매핑이다.
scaled는 시작 시 로봇 손끝 방향을 기준으로 회전 변화만 적용하므로 이 옵션을 사용하지 않는다.

## 4. 빠른 응답에서 변경되는 값

아래 값은 응답 프로필의 기본 파라미터다. 새 URDF IK는 이 위에 관절 제한 감속,
자세 선호와 위치 우선 처리를 적용하며 DLS damping도 특이 자세 근처에서 자동 증가한다.
따라서 `smooth`는 **응답 프로필만** 이전 값으로 바꾼다. 기존 알고리즘 비교는 `--arm-ik legacy`도 필요하다.

| 설정 | `responsive` (scaled/absolute 컨트롤러 기본) | `smooth` (기존) |
| --- | --- | --- |
| 입력 필터 시정수 | 15ms | 45ms |
| 위치·회전 오차 보정 이득 | 10/s | 2.5/s |
| DLS damping (특이 자세에서 IK 안정화) | 0.05 | 0.08 |
| 관절 속도 상한 | 2.5rad/s | 1.5rad/s |
| 관절 가속도 상한 | 20rad/s² | 12rad/s² |

느린 응답 프로필을 교체하는 것이지, 목표에 순간이동시키거나 모든 필터·제한을 없애는 것이 아니다.
관절 위치·기존 actuator 토크 제한, 목표 누적 선행량 0.1rad 제한과 추적 손실 정지는 유지한다.
위 수치는 측정된 Quest 지연 시간이 아니다. 필터·이득을 더 조정해야 한다면
`src/kuavo_isaaclab_scene/teleop/teleop_servo.py`의 프로필이 정의 위치다.

## 5. 여전히 늦거나 어긋나면

| 증상 | 먼저 확인할 항목 |
| --- | --- |
| 이동 거리가 너무 크거나 작음 | scaled의 `position-gain`; 손을 멈춘 뒤에도 차이가 남는지 확인 |
| 목표는 맞지만 팔이 늦게 따라옴 | 시작 로그 `Arm response=responsive`, `[MOTION] target error`, `[PERF] loop` |
| 방향만 다름 | `absolute-orientation`, `arm-orientation-weight`; scaled라면 A 재개 시 기준 방향 |
| 양팔이 갑자기 멈춤 | `[TRACKING]`·`[SAFETY]`; 한쪽 손 또는 머리 추적 손실도 정지 조건 |
| 멈춘 뒤에도 target error가 큼 | 도달 불가능한 자세, 관절 제한, 물체 접촉 여부 |

설정 60Hz인데 실제 loop가 30Hz이면 시뮬레이션 시간도 벽시계보다 느리게 진행된다.
`--control-hz 30`으로 비교할 수 있지만 실제 loop 속도는 다시 확인해야 한다.
렌더링·네트워크 지연은 팔 응답 설정만으로 해결되지 않는다.
[화면·성능 옵션](QUEST3_DISPLAY_AND_PERFORMANCE.md)을 함께 참고한다.

구형 relative/브라우저 매핑의 프레임당 translation 2.5cm·rotation 0.12rad 제한은
`TeleopMappingCfg`에 있으며 scaled/absolute 목표 생성에는 적용되지 않는다.
이 제한과 공통 IK의 관절 속도·가속도 제한을 혼동하지 않는다.
