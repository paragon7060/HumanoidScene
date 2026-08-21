# Kuavo Isaac Sim / Isaac Lab workcell guide

이 문서는 이 프로젝트의 실행, Isaac Sim 수동 배치, 위치·회전·크기 캡처,
Isaac Lab 재스폰 방법을 한곳에 정리한 기준 문서다.

프로젝트 경로:

```bash
cd HumanoidScene
```

저장소는 `src/kuavo_isaaclab_scene` Python package, `scripts` 실행 도구,
`tests`, `docs`, `artifacts`로 분리되어 있다. 루트의 `.sh` 파일은 기존
명령 호환 wrapper이며 실제 구현은 `scripts/`와 package 안에 있다.

## 1. 현재 사용하는 에셋과 좌표 기준

- 랙: `src/kuavo_isaaclab_scene/assets/Rack.usd`
- 박스: `src/kuavo_isaaclab_scene/assets/{Small,Medium,Large,XLarge}Box.usd`, 종류별 2개
- 랙 기준 prim: `/World/envs/env_0/Workcell/Racks/Rack`
- 랙 USD visual: `/World/envs/env_0/Workcell/Racks/Rack/Visual`
- 박스 root prim: `/World/envs/env_0/Workcell/StagingBoxes/<BoxName>`
- 저장 단위: metre, quaternion 순서: `(w, x, y, z)`

박스는 랙 prim 내부에 들어가지 않는다. 8개 박스는 모두
`StagingBoxes` 아래의 독립 articulation이며, 스폰 위치만 랙 기준으로
계산된다. 캡처한 `local_pos`와 `local_rot`도 `Rack` 좌표계 기준이다.
따라서 랙을 나중에 이동하거나 회전해도 박스 배열이 같이 따라간다.
랙 scale은 박스 위치 간격에 반영되지만 박스 자체 scale은 각 박스 값으로
독립 관리된다.

## 2. 기본 실행

Standalone `InteractiveScene`:

```bash
./run_scene.sh --prefill 2
```

Manager-based 환경:

```bash
./run_manager_env.sh --num-envs 1 --steps 100000
```

GUI 실행 시 head, waist, left wrist, right wrist 카메라가 작은 viewport로
열린다. `--headless`에서는 이 창들이 열리지 않는다.

## 3. 코드/CLI로 초기 박스 구성하기

선반 번호는 `1=아래`, `2=중간`, `3=위`다. 박스 이름은 `small`,
`medium`, `large`, `xlarge`이며 각 종류는 최대 2개다.

```bash
./run_scene.sh --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge' \
  --ignore-captured-box-poses
```

`--ignore-captured-box-poses`는 기존 `configs/rack_box_poses.json` 대신 새 자동
배치를 편집용 시작점으로 사용할 때 중요하다. 선반당 자동 배치는 최대
4개이며 앞의 2개는 앞줄, 다음 2개는 뒷줄이다.

JSON 구성도 사용할 수 있다.

```json
{
  "shelves": {
    "1": {"small": 2, "medium": 1},
    "2": ["large", "xlarge"],
    "3": ["medium", "large", "xlarge"]
  }
}
```

```bash
./run_scene.sh --rack-box-layout /absolute/path/rack_layout.json \
  --ignore-captured-box-poses
./run_manager_env.sh --num-envs 1 --steps 100000 \
  --rack-box-layout /absolute/path/rack_layout.json \
  --ignore-captured-box-poses
```

코드 기본값은 `src/kuavo_isaaclab_scene/rack_box_layout.py`의 다음 dictionary로 관리한다.

```python
DEFAULT_RACK_BOX_LAYOUT = {
    1: {"small": 2, "medium": 1},
    2: {"large": 1, "xlarge": 1},
    3: ["medium", "large", "xlarge"],
}
```

자동 박스 목표 크기는 같은 파일의 `BOX_DIMENSIONS_M`에서 metre 단위로
조절할 수 있다. Isaac Sim에서 캡처하면 각 박스 root의 실제 scale이 이
자동값보다 우선한다.

## 4. Isaac Sim에서 랙과 박스를 직접 배치하기

### 4.1 편집용 scene 열기

원하는 박스 인스턴스가 보이도록 초기 구성을 지정하고 기존 캡처는 끈다.

```bash
./run_scene.sh --rack-boxes '1:small*2;2:medium,large;3:xlarge*2' \
  --ignore-captured-box-poses
```

Isaac Sim 창이 열리면 먼저 Timeline을 Pause한다. 실행 중인 rigid body를
강제로 옮기면 physics가 다시 위치를 갱신하거나 박스가 튈 수 있다.

### 4.2 이동·회전·크기 조절

Stage 트리에서 반드시 아래 root prim을 선택한다.

| 대상 | 선택할 prim |
|---|---|
| 랙 전체 이동/회전/크기 | `/World/envs/env_0/Workcell/Racks/Rack` |
| 랙 USD visual | `/World/envs/env_0/Workcell/Racks/Rack/Visual` — 직접 변형하지 않음 |
| Small 0 | `/World/envs/env_0/Workcell/StagingBoxes/SmallBox_0` |
| Small 1 | `/World/envs/env_0/Workcell/StagingBoxes/SmallBox_1` |
| Medium 0/1 | `/World/envs/env_0/Workcell/StagingBoxes/MediumBox_0`, `_1` |
| Large 0/1 | `/World/envs/env_0/Workcell/StagingBoxes/LargeBox_0`, `_1` |
| XLarge 0/1 | `/World/envs/env_0/Workcell/StagingBoxes/XLargeBox_0`, `_1` |

Move/Rotate/Scale 도구로 배치한다. `Body`, flap link, collision mesh 같은
박스 내부 child prim을 옮기지 않는다. 박스 root를 선반 위에 올리고,
선반 경사와 맞도록 회전한 뒤 침투나 프레임 충돌이 없는지 확인한다.

랙 크기를 바꿀 때는 `Rack` root에 Scale을 적용한다. `Rack/Visual`은 항상
local translate `(0,0,0)`, identity rotation, scale `(1,1,1)`로 둔다. 제공된
USD의 측정 native 크기는 X 1.051 m, Y 0.880954 m, Z 2.165 m다. 특정 실측
크기로 맞출 때 필요한 축별 scale은 `목표 크기 / native 크기`다. 예를
들어 X를 0.347 m로 맞추려면 X scale은 약 `0.33016`이다. 축 방향은 실제
Stage의 local gizmo를 기준으로 확인한다.

### 4.3 편집 결과 저장

Isaac Sim에서 **File > Save As**로 저장한다.

```text
artifacts/output/measured_workcell.usda
```

저장 시점에도 Timeline은 Pause 상태로 두는 것이 좋다. 저장 후 Isaac Sim을
닫으면 캡처 과정에서 GPU 메모리를 중복 점유하지 않는다.

## 5. 저장한 USD에서 위치·회전·크기 캡처하기

### 5.1 랙/로봇/컨베이어/펜스 위치 캡처

랙 또는 다른 workcell group을 움직이거나 랙 scale을 바꿨다면 실행한다.

```bash
./capture_layout.sh artifacts/output/measured_workcell.usda
```

`configs/workcell_layout.json`은 robot, rack,
conveyor, fence, button station의
월드 position/rotation과 랙 scale을 저장한다.

### 5.2 랙 위 박스 자동 감지 및 캡처

```bash
./capture_rack_box_poses.sh artifacts/output/measured_workcell.usda
```

랙 내부에 놓인 박스만 자동 감지하여 프로젝트 루트의
`configs/rack_box_poses.json`에 다음 값을 저장한다.

- `local_pos`: `Rack` anchor 기준 위치
- `local_rot`: `Rack` anchor 기준 quaternion `(w, x, y, z)`
- `scale`: 박스 root의 실제 effective scale
- `shelf`: local Z와 가장 가까운 선반 번호

랙 bounds 밖 박스까지 명시적으로 캡처하려면 이름을 지정한다.

```bash
./capture_rack_box_poses.sh artifacts/output/measured_workcell.usda \
  --boxes SmallBox_0 MediumBox_1 LargeBox_0
```

별도 파일로 보관하려면:

```bash
./capture_rack_box_poses.sh artifacts/output/measured_workcell.usda \
  --output layouts/shift_a_boxes.json
```

기본 `configs/rack_box_poses.json`은 1개 이상 랙 위 박스를 포함해야 한다. 랙에서 내린
박스는 자동 감지 대상에서 빠지고 바닥 staging 기본 위치에 스폰된다.

## 6. 캡처 결과로 다시 스폰하기

패키지 `configs/rack_box_poses.json`이 있으면 별도 옵션 없이 두 환경이
자동으로 읽는다.

```bash
./run_scene.sh
./run_manager_env.sh --num-envs 1 --steps 100000
```

다른 캡처 파일을 선택하려면:

```bash
./run_scene.sh --rack-box-poses /absolute/path/shift_a_boxes.json
./run_manager_env.sh --num-envs 1 --steps 100000 \
  --rack-box-poses /absolute/path/shift_a_boxes.json
```

캡처를 일시적으로 끄고 CLI/dictionary 자동 배치만 확인하려면:

```bash
./run_scene.sh --ignore-captured-box-poses \
  --rack-boxes '1:small*2;2:medium,large;3:xlarge*2'
```

CLI/JSON 자동 구성과 캡처 파일을 동시에 주면 캡처에 기록된 인스턴스의
position, rotation, scale, shelf가 우선하며 기록되지 않은 인스턴스는
자동 구성 값을 사용한다.

## 7. 배치를 다시 수정하는 반복 작업

1. 기존 캡처 결과로 `./run_scene.sh`를 실행한다.
2. Timeline을 Pause한다.
3. 박스 root 또는 랙 root를 수정한다.
4. 새 USD/USDA로 Save As한다.
5. 랙도 바뀌었으면 `capture_layout.sh`를 다시 실행한다.
6. `capture_rack_box_poses.sh`를 다시 실행한다.
7. `./run_scene.sh` 또는 manager-based 환경을 재실행한다.

박스만 수정했다면 5번은 생략해도 된다. 랙만 이동·회전하고 상대적인
박스 배열은 그대로 유지하려면 `capture_layout.sh`만 다시 실행해도 된다.

## 8. 박스 flap friction

고정값:

```bash
./run_scene.sh --flap-static-friction 0.40 \
  --flap-dynamic-friction 0.25
```

Standalone 시작 시 박스/flap별 randomization:

```bash
./run_scene.sh --randomize-flap-friction \
  --flap-static-friction-range 0.15 0.65 \
  --flap-dynamic-friction-range 0.08 0.45
```

Manager-based 환경은 reset마다 randomization이 기본 활성화된다.

```bash
./run_manager_env.sh --num-envs 8 --steps 100000 \
  --flap-static-friction-range 0.15 0.65 \
  --flap-dynamic-friction-range 0.08 0.45
```

결정론적 manager-based 실행은 `--no-randomize-flap-friction`을 사용한다.
PhysX 제약 때문에 dynamic 값은 static 값 이하로 clamp된다.

## 9. task logic과 버튼 물리 확인

Standalone oracle demo는 기존 6개 tote용 검증이다. 캡처 파일이 생긴 뒤에는
반드시 캡처를 무시해서 실행한다.

```bash
./run_scene.sh --auto-demo --prefill 2 --ignore-captured-box-poses
```

실제 prismatic plunger 이동, gated completion, spring return을 확인하려면:

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode performance \
  --verify-button --steps 300 --ignore-captured-box-poses
```

수동 캡처 박스를 사용하는 일반 실행에서는 캡처된 `shelf`가 있는 인스턴스만
task box로 등록되고, 모든 task box가 컨베이어에 놓인 뒤에만 초록 버튼이
유효하다. 자동 demo는 로봇 manipulation policy가 아니라 task state machine
검증용이다.

## 10. 화면/파일 확인용 명령

한 프레임 이미지 저장:

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode quality \
  --steps 1 --screenshot artifacts/output/workcell.png
```

composed stage 저장:

```bash
./run_scene.sh --headless --device cuda:0 --steps 1 \
  --save-stage artifacts/output/workcell.usda
```

옵션 전체 확인:

```bash
./run_scene.sh --help
./run_manager_env.sh --help
./capture_layout.sh --help
./capture_rack_box_poses.sh --help
```

사용자가 직접 시각 확인할 항목:

- 박스 바닥과 선반 표면 사이의 침투 또는 큰 간격
- 박스 flap과 랙 프레임의 초기 충돌
- 경사 선반 방향과 박스 rotation의 일치
- rack scale 변경 후 박스 간격과 프레임 여유
- 양손 wrist camera와 head/waist camera 시야

## 11. 주의사항과 문제 해결

- 기본 `configs/rack_box_poses.json`이 있는데 새 `--rack-boxes`가 반영되지 않으면
  `--ignore-captured-box-poses`를 사용한다.
- 자동 캡처가 0개라면 박스 root가 실제로 랙 bounds 안에 있는지 확인하거나
  `--boxes NAME ...`으로 명시한다.
- 랙과 박스가 재실행 후 어긋나면 같은 저장 stage에서 두 캡처 스크립트를
  다시 실행한다.
- scale이 중복 적용된 것처럼 보이면 child mesh가 아니라 rack/box root만
  조절했는지 확인한다.
- `Rack/Visual`에 non-zero transform이 보이면 이전 구조로 저장된 stage일
  수 있다. 새 scene을 연 뒤 `Rack` root만 편집하고 다시 저장·캡처한다.
- GUI Isaac Sim과 headless 캡처를 동시에 실행하면 VRAM 부족으로 다른
  Electron 앱까지 종료될 수 있다. 편집 창을 닫은 뒤 캡처한다.
- package `scene.py`는 시각 확인과 task scheduler용이고, `manager_env.py`는 학습 및
  domain randomization용이다. 두 환경은 동일한 layout/pose JSON을 읽는다.

## 12. 관리 원칙

- workcell anchor 변경: `configs/workcell_layout.json`
- 랙 위 수동 박스 pose/scale 변경: `configs/rack_box_poses.json`
- 자동 선반 구성 변경: package `rack_box_layout.py` 또는 별도 layout JSON
- flap friction 기본값/범위 변경: package `box_flap_friction.py`
- 사용법 변경 시 이 문서를 먼저 갱신하고 README에는 요약과 링크만 둔다.

Isaac Lab 런타임에서는 다음 key를 사용한다.

```python
# Rack anchor: world position/rotation/scale의 기준 XFormPrim
rack = scene["rack"]
rack_pos_w, rack_quat_w = rack.get_world_poses()

# Rack.usd child visual: local pose는 항상 identity
rack_visual = scene["rack_visual"]

# 개별 박스는 Articulation이므로 tensor world pose에 직접 접근 가능
small_box_0_pos_w = scene["small_box_0"].data.root_pos_w
small_box_0_quat_w = scene["small_box_0"].data.root_quat_w
```

다중 환경에서 박스의 환경-local 위치가 필요하면
`box.data.root_pos_w - scene.env_origins`를 사용한다.
