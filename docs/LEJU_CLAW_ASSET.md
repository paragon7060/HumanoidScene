# 독립 Leju two-finger claw asset

대회 S52 모델과의 차이는 [OpenLET gripper 비교](LEJU_CLAW_COMPARISON.md)에 정리했다.

## 추출 범위

`src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/`에 독립 패키지가 있다.
좌우 URDF, 해당 mesh, 실행용 USD 및 configuration layer, 관성 추정값,
장착/구동/접촉 설정 `config.json`, 출처 `PROVENANCE.md`를 포함한다.
wheel 패키지에도 기존 `assets/**/*` 규칙으로 포함된다.

각 손은 14 links / 13 tree joints이며 D405 카메라 branch까지 포함한다.
원점은 `l/r_twofinger_base`이다. 로봇 몸체·팔·head camera는 포함하지 않는다.
손마다 두 jaw의 four-bar closure hinge가 USD에 추가되어 있다.
bar_1 두 개만 구동하며 bar_3/4는 passive이고 reset 때 폐루프 해로 초기화한다.
URDF만 import하면 closure가 없으므로 반드시 USD finalization이 필요하다.
색·형상은 원본 그대로이고, 각 손 패키지는 D405를 포함해 **1.000 kg**이다.
기존 0.740 kg 추정 모델의 링크별 질량 비율과 CoM을 유지하며 질량과 관성을
동일 비율로 조정했다. 실측 관성값은 아니며 S200062 원본 손 질량은 변경하지 않는다.
URDF·USD와 runtime spawn에 같은 값을 적용하며, 반복 적용해도 질량이 누적되지 않는다.

기존 S200062와 S56 실행 경로는 그대로 유지한다.
`--robot-model s63 --gripper leju-twofinger`는 양손 claw를 합친 별도 S63 variant를
선택하며, `--robot-model s63 --gripper none`은 기본 S63을 선택한다.
기본 S63은 `l/r_hand_pitch.STL`을 사용한다. twofinger variant는 준비된
`kuavo_s63.urdf`를 입력으로 생성하되, 손목 visual만 파생
`l/r_hand_pitch_wrist_only.STL`로 교체한다. S63 CAD에 포함된 카메라와 브래킷
3개 부품을 제거해, 손마다 Leju D405 카메라와 마운트 하나만 남긴다.

## S63에서 바로 실행

```bash
./run_scene.sh --robot-model s63 --gripper leju-twofinger
./run_manager_env.sh --robot-model s63 --gripper leju-twofinger --num-envs 1
```

`scene["robot"]` 안에 좌우 claw 관절이 들어가는 integrated 방식이다.
외장 hand를 또 spawn하지 않으며 기존 양손 action 채널로 제어한다.
원본 S63 STL은 보존하며, variant의 손목 visual에서만 중복 카메라 형상을 제거한다.
EEF, host 프레임, 관절·관성·충돌 설정은 유지한다.
head camera는 S63 기준, wrist camera는 추출한 좌우 D405의 물리 body 기준이다.
mesh 겹침이나 파지 성능은 아직 시각/접촉 검증하지 않았다.

이 preset은 S63 전용이다. S200062는 기존 `s200062_integrated`를 사용한다.
S200062용 closed-TCP 보정값은 S63에 자동 적용하지 않으며 공식 EEF를 유지한다.
RL robot factory도 preset을 허용하지만 task의 모델 전용 초기 상태·보정·학습
config는 따로 준비해야 한다. 전용 S200062 four-box baseline의 제한은 유지한다.

장착 transform은 `assets/leju_claw_two_finger/config.json`의
`sides.left/right.source_mount_xyz_m` 및 `source_mount_rpy_rad`에서 수정한 뒤:

```bash
bash scripts/build_s63_twofinger.sh
```

이 명령은 S63+claw variant만 재생성한다. `grippers.json`은 preset 이름을 이
패키지에 연결하는 registry일 뿐이다. 구동 gain, open/close command, host별 보정,
접촉 재질, distal pad와 force control은 모두
`assets/leju_claw_two_finger/config.json`에서 수정한다. runtime 값은 시뮬레이터를
재시작하면 적용되고, USD에 bake되는 형상·물리 값은 아래 빌드 명령으로 재생성한다.
구동 각도는 검증된 범위를 벗어나지 않는다.

기본 닫기 힘은 한 손의 양쪽 손가락 합계 50 N(각 25 N)이며
`--gripper-close-force`로 변경한다. RL 학습·평가와 RL reward debug는 binary
`0=open`, `1=close` 위치 명령과 기존 PD를 유지하면서 sensor-free geometric
feedforward를 더한다. 이 값은 측정 힘이 아니라 linkage에서 환산한 N-equivalent이다.
빈 gripper가 닫힘 mechanical stop에 도달하면 보조 토크를 끊는다.

일반 VR dataset 수집도 RL 학습·평가와 RL reward debug와 같은 sensor-free geometric
feedforward를 사용한다. 따라서 binary 명령, position mapping, target filter, 위치 PD와
50 N-equivalent 보조가 모든 경로에서 동일하다. `--gripper-close-force 0`은 모든 경로의
힘 보조를 끄고 위치 개폐만 사용한다.

각 finger의 끝 20 mm에는 18 mm 폭, 2 mm 두께의 평평한 전용 contact pad가 있다.
pad 접촉면은 원본 CAD convex hull보다 0.5 mm 안쪽으로 돌출되어 얇은 flap을 잡을 때
곡면 hull보다 먼저 접촉한다. 원본 hull은 finger 나머지 부분의 충돌 보호용으로 유지한다.

전체 구성 요약은 [Gripper 구성](GRIPPER.md)에 있다.

이 pad는 기본적으로 강체가 아니라 얇고 부드러운 패드로 동작한다. PhysX compliant contact를
사용해 하중을 받으면 눌리고, 눌린 깊이에 비례하는 힘을 돌려준다. 강체 pad가 강체
box flap을 누를 때처럼 접촉이 on/off로 풀리면서 수직력이 순간적으로 0이 되는 현상을
줄이는 것이 목적이다. 스프링 상수는 pad 자체의 압축 강성 `E x 면적 / 두께`이며
`config.json`의 `contact.distal_pad.compliance`가 유일한 출처다. 기본값은
E = 280 kPa(무른 고무/실리콘), 50 400 N/m, damping 250 Ns/m이다. PhysX는 이 스프링을
contact point마다 적용하므로 평면끼리 만드는 4점 patch에서는 50 N 파지 시 약
0.25 mm 눌린다. 0.6 mm 뒤의 원본 hull이 더 깊이 들어가는 것을 막는 backstop이다.
마찰계수는 기존 finger 값을 그대로 쓰며 pad 때문에 달라지지 않는다.

이전의 강체 pad도 그대로 쓸 수 있다. `--gripper-pad rigid`를 주면 같은 위치·크기의
slab이 finger 재질을 공유하는 강체 면으로 만들어지고, `--gripper-pad soft`(기본)는
연성 패드를 쓴다. preset의 `finger_contact.soft_pad`로 기본값을 바꿀 수도 있다.

접촉 모델은 `usd.py` 한 곳에만 있다. 재질과 collider는
`author_claw_jaw_contact()`가, soft pad와 그 재질은 `author_claw_distal_pads()`가
만든다. 독립 claw와 S63 조합의 `author_claw_contact()`, S200062/S56 통합 경로의
`author_integrated_claw_contact()`가 모두 이 두 함수를 호출하므로 한쪽만 수정되어
갈라질 일이 없다.

host별로 다른 것은 넘겨주는 값뿐이다. 패키지 claw와 S63은 실제 `collisions` mesh를
가지고 있지만 S200062/S56 donor USD는 `collisions` scope가 비어 있어 `visuals`
mesh를 collider로 쓴다. 또 이 donor들의 wrist link에는 URDF가 준 원통 collider가
있어 `replace_colliders`로 끄고 mesh hull로 대체한다. S63은 wrist를 건드리지 않아
원통 collider가 그대로 남는다. host URDF의 지름 10 mm `zarm_*7_end_effector` 기준
sphere는 좌표 기준일 뿐 물리 부품이 아니므로 spawn 시 collision을 비활성화한다.

## 재생성

저장소 루트에서:

```bash
# URDF / mesh / metadata만 추출: Isaac Sim을 시작하지 않음
conda activate env_isaaclab_232
python scripts/extract_leju_claw.py

# 좌우 USD까지 변환·closure/contact 적용: GUI 없이 asset만 빌드
bash scripts/build_leju_claw.sh
```

추출은 기존 donor 파일을 변경하지 않는다. 기존 `config.json`의 action, actuator,
contact, runtime preset과 force control을 보존하므로 빌드 스크립트의 상수까지
따로 수정할 필요가 없다. contact만 USD에 재적용할 때는:

```bash
python scripts/finalize_twofinger_usd.py \
  src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/usd/left/leju_claw_left.usd \
  --sides l --claw-config src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json
# 오른손은 경로 right와 --sides r로 변경
```

## IsaacLab에서 독립 articulation 사용

AppLauncher 실행 후 scene config에 추가한다:

```python
from kuavo_isaaclab_scene.robots.claw_assets.isaaclab import make_claw_cfg

left_claw = make_claw_cfg("left", "{ENV_REGEX_NS}/Grippers/Left", pos=(0, 0, 1))
right_claw = make_claw_cfg("right", "{ENV_REGEX_NS}/Grippers/Right", pos=(0, -.3, 1))
```

기본 `fix_base=True`는 단독 테스트용이다. scene key를 `left_claw`로 추가하면
`scene["left_claw"].data.root_pos_w`가 world root 위치이며 root와 베이스가
같은 원점이다. `Grippers` parent는 먼저 생성되어 있어야 한다.
독립 USD는 free-base이고 cfg가 단독 테스트의 world fixed joint를 만든다.
로봇에 물리 장착할 경우 `fix_base=False`로 사용하고 별도 mount joint 및 초기
world pose를 일치시켜야 한다. **cfg만 추가해서 손목을 따라가게 되지는 않는다.**

구동 targets는 simulator-independent API로 계산한다:

```python
from kuavo_isaaclab_scene.robots.claw_assets import load_claw_asset
claw = load_claw_asset("left")
targets = claw.motor_positions(+1)  # +1 open, -1 closed, 0 half open
reset = claw.initial_positions(+1) # 두 driver + 네 passive joints
# find_joints(claw.motor_names, preserve_order=True) 후 driver targets만 전송
```

actuator gain은 `config.json`의 `actuator`에서 수정하며 `make_claw_cfg()`가 읽는다.
passive joint에 position target을 주거나 stiffness를 추가하지 않는다.
finger friction 20/16은 기존 프로젝트 값이며 매우 높다. 접촉 검증 후 튜닝한다.

## 패키지 코드 위치

Leju two-finger 관련 구현은 `robots/claw_assets/` 아래에서 관리한다.

| 파일 | 역할 |
|---|---|
| `package.py` | 단일 `config.json` 로딩, asset 경로, URDF branch 조합 |
| `linkage.py` | four-bar 폐루프 기구학과 USD joint 작성 |
| `geometry.py` | 손가락 간격과 접촉 형상 계산 |
| `force.py` | 50 N force servo와 토크 변환 |
| `vr.py` | Quest/RL debug용 contact sensor와 force-close action 구성 |
| `usd.py` | friction, flat pad, EEF marker 비활성화와 USD inertial 반영 |
| `isaaclab.py` | 독립 claw 및 S200062/S56/S63 runtime 연결 |

`robots/twofinger_linkage.py`, `robots/twofinger_geometry.py`,
`robots/gripper_force.py`, `robots/vr_gripper_force.py`는 기존 import를 위한 얇은
호환 wrapper다. 새 구현이나
수치 설정을 이 파일에 추가하지 않는다. 세 로봇 preset은 `grippers.json`에서
각각 `package_preset`만 선택하므로 공통 변경은 package `config.json` 한 곳에서,
장착/응답 보정만 `runtime_presets`의 해당 host 항목에서 수정한다.

## S63 등 다른 로봇 URDF에 합칠 때

`robots/claw_assets/package.py`의 `append_claw_branch()`는 XML tree에 hand와
attachment fixed joint를 추가한다. host 파일을 쓰거나 visual을 제거하지 않는다.

```python
import xml.etree.ElementTree as ET
from pathlib import Path
from kuavo_isaaclab_scene.robots.claw_assets import append_claw_branch

tree = ET.parse("src/kuavo_isaaclab_scene/assets/kuavo_s63/urdf/kuavo_s63.urdf")
output = Path("/tmp/s63_claw.urdf")  # 별도 variant; 공식 모델에 덮어쓰지 않음
append_claw_branch(tree.getroot(), side="left", parent_link="zarm_l7_link",
                   output_urdf=output, xyz=(0, -.0005, -.041))
append_claw_branch(tree.getroot(), side="right", parent_link="zarm_r7_link",
                   output_urdf=output, xyz=(0, +.0005, -.041))
# host mesh 경로도 output 위치 기준으로 수정한 다음 tree.write(output, ...)
```

위 mount는 **S200062 donor 값**이며 실제 S63에 맞는지 검증된 값은 아니다.
xyz(m)/rpy(rad)는 wrist `zarm_*7_link` 기준이다. EEF 기준 값으로 혼용하지 않는다.
S63 공식 wrist visual과 donor claw가 겹치는지도 별도로 확인해야 한다.
helper는 원본 visual/EEF를 임의로 바꾸지 않으므로 visual 대체가 필요하면
별도 variant에서 명시적으로 결정한다.

합친 URDF는 fixed links를 merge하지 않고 USD로 변환한다. 양손을 합친 USD는
`scripts/finalize_twofinger_usd.py <새USD>`로 네 closure를 추가해야 하며,
`robots/claw_assets/usd.py`의 `author_claw_contact()`를 좌우 각각 적용하면
패키지 접촉 설정도 반영된다. S63 variant 선택 및 물리 D405 camera 연결은
위 `leju-twofinger` 실행 경로에 구현되어 있다. IK 기준은 공식 S63 EEF다.
