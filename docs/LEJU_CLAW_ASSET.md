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
선택하며, `--robot-model s63 --gripper none`은 공식 S63 그대로다.
twofinger variant는 손목 visual만 파생 `l/r_hand_pitch_wrist_only.STL`로 교체한다.
S63 `hand_pitch.STL` CAD에 포함된 카메라와 브래킷 3개 부품을 제거해,
손마다 Leju D405 카메라와 마운트 하나만 남긴다.

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

이 명령은 S63+claw variant만 재생성한다. `grippers.json`의 integrated preset
`robot_mount_pos`를 수정해서 장착 위치가 바뀌지는 않는다. 구동 gain, open/close
command와 finger friction은 `configs/grippers.json`의 `leju-twofinger`에서
수정한다. runtime 구동·마찰 설정은 시뮬레이터를 재시작하면 적용된다.
구동 각도는 검증된 범위를 벗어나지 않는다.

VR 수집과 RL reward debug에서는 닫기 명령을 기본 50 N 압착력 목표로 사용한다.
이는 한 손의 양쪽 손가락 합계(각 25 N)이며 `--gripper-close-force`로 변경한다.
접촉 전부터 닫기 방향 토크를 공급하고 접촉 후 센서 피드백으로 힘을 조절한다.
열기는 같은 힘을 반대 방향 토크로 환산하며, 닫힘/열림 끝에서 토크를 끊는다.
개폐 모두 위치 stiffness를 끄고 implicit damping으로 속도를 억제한다. 기존 토크
예산의 절반씩을 외부 토크와 damping drive에 할당한다. reset 때 원래 구동 설정을
복구한다. `--gripper-close-force 0`은 기존 위치
개폐 동작을 사용한다. 이 옵션은 독립 claw asset이나 학습용 policy에 자동으로
힘 제어를 추가하지 않는다.

## 재생성

저장소 루트에서:

```bash
# URDF / mesh / metadata만 추출: Isaac Sim을 시작하지 않음
conda activate env_isaaclab_232
python scripts/extract_leju_claw.py

# 좌우 USD까지 변환·closure/contact 적용: GUI 없이 asset만 빌드
bash scripts/build_leju_claw.sh
```

추출은 기존 donor 파일을 변경하지 않는다. `config.json`을 수동 조절한 뒤
다시 `build_leju_claw.sh`를 실행하면 추출 기본값으로 갱신되므로, 영구 기본값은
`scripts/extract_leju_claw.py`에서 수정한다. contact만 USD에 재적용할 때는:

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
