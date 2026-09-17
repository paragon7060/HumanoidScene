# Gripper 구성

Leju two-finger claw가 어떻게 조립되고, 어디를 고치면 무엇이 바뀌는지 한 곳에
정리한 문서다. 세부 절차는 각 항목에서 전용 문서로 연결한다.

## 단일 출처

구현은 `src/kuavo_isaaclab_scene/robots/claw_assets/` 패키지 하나뿐이고,
수치는 `src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json`
한 파일이 기준이다. `configs/grippers.json`의 세 항목은 이름을 패키지
`runtime_presets`에 연결하는 별칭이며 같은 값을 복사해 두지 않는다.

```
assets/leju_claw_two_finger/
  config.json              수치 기준: actuator, contact, soft pad, force, host 캘리브레이션
  inertial_estimates.json  링크별 질량/CoM/관성 (추정값, 제조사 실측 아님)
  urdf/, meshes/, usd/     좌우 독립 claw 형상
robots/claw_assets/
  package.py    경로, 기본 파지력, URDF 합성(append_claw_branch)
  linkage.py    4절 링크 기구학과 USD closure joint
  geometry.py   오프라인 jaw 간격/접촉 형상 계산
  force.py      힘 feedforward와 접촉센서 되먹임 서보
  vr.py         Quest/RL debug용 접촉센서와 force-close action 구성
  usd.py        접촉 모델 전체: 관성, collider, 재질, soft pad, marker 비활성화
  isaaclab.py   Isaac Lab spawn, host별 링크·mesh scope 전달
```

## host별 장착 방식

| host | preset | claw가 붙는 방식 | collider mesh scope | 진입점 |
|---|---|---|---|---|
| S63 | `leju-twofinger` | 공식 mesh를 수정하지 않고 URDF에 branch를 합성해 별도 USD로 변환 | `collisions` | `spawn_s63_twofinger_robot` |
| S200062 | `s200062_integrated` | 원본 로봇 USD에 이미 포함 | `visuals` | `author_integrated_claw_contact` |
| S56 | `s56_twofinger` | S200062 hand rig를 이식 | `visuals` | `author_integrated_claw_contact` |
| 독립 claw | - | 손만 단독 spawn | `collisions` | `spawn_claw` |

mesh scope가 갈리는 이유는 의도가 아니라 에셋 차이다. 패키지 claw와 S63은 URDF에
collision 형상이 있어 `collisions` mesh를 쓰지만, S200062/S56 donor USD는
`collisions` scope가 비어 있어 `visuals` mesh가 유일한 손 형상이다.

접촉 모델 자체는 host와 무관하게 `usd.py`의 두 함수에만 있다.
`author_claw_jaw_contact()`가 재질과 collider를, `author_claw_distal_pads()`가
soft pad와 그 재질을 만든다. 위 네 진입점이 모두 이 함수를 호출하므로 한쪽만
고쳐져서 갈라질 수 없다. host가 넘기는 것은 링크 이름과 mesh scope뿐이다.

## 기구부

한 손에 4절 링크 두 벌이 있고, 구동 관절은 `{side}_f_bar_1_joint`와
`{side}_b_bar_1_joint` 두 개다. 나머지 관절은 USD closure joint로 닫힌
링크를 이룬다. 명령 백분율과 관절 목표의 대응은 실측 lookup이며
[S63 개폐 제어](S63_GRIPPER_CONTROL.md)에 있다. 형상·재생성·장착 원점은
[독립 Leju claw asset](LEJU_CLAW_ASSET.md)을 본다.

## 접촉 모델

collider는 convex hull이고 contact offset 2 mm, rest offset 0, speculative CCD를
쓴다. 재질은 세 가지다.

| 재질 | 대상 | 값 |
|---|---|---|
| `FingerContactMaterial` | finger hull | 마찰 20 / 16, average |
| `HandContactMaterial` | claw 하우징, 손목 | 마찰 1.0 / 0.8 |
| `SoftPadContactMaterial` | 끝단 soft pad | finger와 같은 마찰 + compliant contact |

finger 끝 20 mm의 파지 평면에는 18 mm 폭, 2 mm 두께의 전용 pad가 있다. 같은 위치,
같은 크기의 이 slab을 두 가지 모델 중 하나로 만들 수 있다.

| 모드 | 재질 | 거동 | 선택 |
|---|---|---|---|
| soft (기본) | `SoftPadContactMaterial` | PhysX compliant contact로 눌림, 화면에 보임 | `--gripper-pad soft` |
| rigid | `FingerContactMaterial` | 이전과 같은 강체 면, invisible | `--gripper-pad rigid` |

soft 모드의 스프링 상수는 pad 자체의 압축 강성 `E x 면적 / 두께`이고 E = 280 kPa에서
50 400 N/m, damping은 250 Ns/m이다. PhysX가 contact point마다 적용하므로 평면끼리
만드는 4점 patch에서 50 N 파지 시 약 0.25 mm 눌리고, 0.6 mm 뒤의 원본 hull이
backstop이 된다. 목적은 마찰을 올리는 것이 아니라, 강체끼리 접촉이 on/off로 풀리면서
수직력이 순간적으로 0이 되는 것을 막는 데 있다. 두 모드는 pad의 위치와 크기가 같아서
개폐 캘리브레이션은 어느 쪽에서도 달라지지 않는다.

모드는 실행 시 `--gripper-pad`로 고르고, 옵션을 주지 않으면 preset의
`finger_contact.soft_pad` 값(기본 soft)을 쓴다. 두 경로 모두 spawn 시 결정되므로
재빌드는 필요 없다. 실행 로그의 `[CONTACT]` 줄에 어느 모드가 적용됐는지 나온다.

손목 링크에는 URDF가 준 지름 80 mm, 길이 140 mm 원통 collider가 있는데 실제 손목
형상(57 x 81 x 96 mm)보다 크고 끝보다 40 mm 더 튀어나온다. 세 host 모두 이 원통을
끄고 손목 mesh hull로 대체한다. 좌표 기준용 지름 10 mm `zarm_*7_end_effector`
sphere도 물리 부품이 아니므로 spawn 시 collision을 끈다.

## 제어

RL 학습·평가와 RL reward debug는 binary 위치 명령(`0=open`, `1=close`)에
sensor-free feedforward를 더한다. VR 데이터 수집은 접촉센서 되먹임을 쓰는 별도
경로로, 접촉 전부터 닫기 방향 힘을 공급하고 접촉 후 측정값으로 조절한다. 기본
파지력은 한 손 50 N이며 `--gripper-close-force`로 바꾸고, 0을 주면 두 경로 모두
위치 개폐만 쓴다. 자세한 내용은 [RL binary gripper](RL_BINARY_GRIPPER.md)와
[S63 개폐 제어](S63_GRIPPER_CONTROL.md)에 있다.

## TCP

파지 기준점은 네 손가락 링크 로컬 좌표로 측정한 접촉점이고
`configs/grasp_reference_points.json`에 있다. 값이 손가락 링크 기준이므로 같은 claw를
쓰는 host는 모두 같은 값을 쓰고, 닫힘 TCP는 각 host의 URDF FK로 계산한다. S200062와
S63 모두 `zarm_*7_end_effector`에서 약 55.5 mm 아래가 닫힘 jaw 중점이다. 이 TCP는
IK, reward, observation, 기록에서 같은 정의를 쓴다. Leju two-finger가 아닌 hand는
보정점이 없으므로 task spec의 fallback offset을 쓴다. 측정 절차는
[End-effector 기준점](ENDEFFECTOR_CENTER.md)과
[desktop 보정](GRASP_DESKTOP_CALIBRATION.md)에 있다.

## 무엇을 고치면 무엇이 바뀌나

| 하고 싶은 일 | 고칠 곳 | 재빌드 |
|---|---|---|
| 파지력 기본값 | `config.json` `force_control.close_force_n` | 불필요 |
| pad 모델(soft/rigid) | `--gripper-pad` 또는 preset `finger_contact.soft_pad` | 불필요 |
| 힘 서보 응답 | `config.json` `force_control`의 gain/ramp | 불필요 |
| pad 물성(무르기) | `config.json` `contact.distal_pad.compliance` | 불필요 |
| pad 크기·위치 | `config.json` `contact.distal_pad.size_m`, `center_m` | 불필요 |
| 마찰계수 | `config.json` `contact.finger_*`, `housing_*` | 불필요 |
| 개폐 캘리브레이션 | `config.json` `runtime_presets.*.position_mapping` | 불필요 |
| 관절 gain | `config.json` `actuator` | `build_leju_claw.sh` |
| 링크 질량·관성 | `inertial_estimates.json` | 불필요 |
| mesh·기구 형상 | `scripts/extract_leju_claw.py` 후 재빌드 | 필요 |
| 파지 기준점(TCP) | `configs/grasp_reference_points.json` | 불필요 |

관성, collider, 재질, soft pad는 spawn마다 다시 authoring되므로 프로세스만 다시
띄우면 반영된다. USD에 baked되는 것은 mesh와 closure joint drive다.

```bash
conda activate env_isaaclab_232
python scripts/extract_leju_claw.py    # URDF/mesh/metadata만 추출
bash scripts/build_leju_claw.sh        # 좌우 독립 claw USD
bash scripts/build_s63_twofinger.sh    # S63 + claw 합성 USD
```

## 관련 문서

- [Gripper 설정과 선택](GRIPPER_CONFIGURATION.md): preset 선택과 실행 옵션
- [독립 Leju claw asset](LEJU_CLAW_ASSET.md): URDF/USD 재생성, 장착 원점, 재사용 API
- [S63 개폐 제어](S63_GRIPPER_CONTROL.md): 명령 백분율과 관절 목표 lookup
- [RL binary gripper](RL_BINARY_GRIPPER.md): 학습용 이산 개폐 action
- [S63 / Leju claw 상태](S63_LEJU_CLAW.md), [claw 비교](LEJU_CLAW_COMPARISON.md)
- [End-effector 기준점](ENDEFFECTOR_CENTER.md)
