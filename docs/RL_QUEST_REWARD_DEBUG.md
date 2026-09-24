# Quest로 움직이며 RL reward 확인

기존 Quest 수집기에 `--rl-reward-debug`만 추가한다. CloudXR Runtime, HTTPS 주소,
Quest CONNECT 절차는 [기존 수집기](QUEST_COLLECTOR_SETUP.md)와 같다.
이 옵션이 없으면 기존 수집 동작은 그대로다. **옵션이 있을 때는 데이터 저장이나
정책 학습 대신 현재 flap-pick RL 환경 1개를 수동 조작한다.**

Whole-body mode 1/2 preset의 IK orientation weight는 일반 collector와 같은 `0.5`다.
필요하면 `--arm-orientation-weight`를 명시해 덮어쓸 수 있다.

관절 목표와 실제 sim 응답을 보정 자료로 기록하려면 `--joint-response-log PATH.jsonl`을 추가한다. 이미지 데이터셋과 독립적으로 동작하며 절차는 [VR 관절 응답 기록](VR_JOINT_RESPONSE_CALIBRATION.md)을 따른다.

S63의 기본 `--dynamics-profile auto`는 몸통과 양팔 모두에 live inverse-dynamics를
사용한다. 시작 로그의 `dynamics_profile=s63-body-id`와 `inverse dynamics on 18 joints`로
확인한다. 몸통만 예전 gravity-PD로 되돌려 비교하려면 `--dynamics-profile s63-arm-id`,
전체를 gravity-only로 되돌리려면 `--dynamics-profile gravity`를 지정한다.

Closed two-finger gripper의 닫기 명령은 기본 `--gripper-close-force 50`으로
한 손의 양쪽 손가락 합계 50 N-equivalent(각 25 N)를 위치 PD에 더한다. reward
debug는 학습과 같은 binary action(`0=open`, `1=close`), 실물 보정 position mapping,
target filter, PD gain과 sensor-free geometric feedforward를 그대로 사용한다. 접촉
센서를 힘 제어 루프에 추가하지 않으므로 HUD의 `GRIP L/R`과 `[GRIP ASSIST]` 로그는
측정 접촉력이 아니라 명령한 등가 힘을 표시한다. 실제 힘은 접촉 형상, 마찰과 solver
compliance에 따라 달라질 수 있다. 빈 gripper가 닫힘 mechanical stop에 도달하면 불필요한
하중을 끊는다. `--gripper-close-force 0`은 보조 토크를 끄고 binary 위치 PD만 사용한다.

일반 VR dataset 수집도 같은 sensor-free gripper backend를 사용하므로 binary 명령,
position mapping, target filter, 위치 PD와 50 N-equivalent 보조가 reward debug와 같다.

## 박스 인출 → 컨베이어 놓기 검사

`--rl-task pick_place --rl-reward-debug 1`은 한 박스를 처음부터 컨베이어에
내려놓는 연속 작업이다. 기존 `pick`은 lift 성공에서 종료된다.
두 모드 모두 기본 task box는 `small_box_0`이며, 전용 고정 pose를 사용해 기존 task
위치인 중간 선반(2번)에 생성한다. 다른 캡처 박스를 검사할 때만
`--rl-box medium_box_0`처럼 지정하며, 선택한 box가 현재 rack layout 또는 캡처 pose에서
선반 위에 있어야 한다.

```bash
./quest_collector.sh collect \
  --robot-model s63 --gripper leju-twofinger \
  --controller-mapping absolute --absolute-orientation downward \
  --arm-response responsive \
  --rl-reward-debug 1 --rl-task pick_place \
  --no-rl-obstacle-collision \
  --no-rack-rollers \
  --no-quest-camera-overlay --no-camera-preview \
  --no-wrist-cameras --no-head-camera
```

기본 실험은 `configs/rl_pick_place.py`이며 별도 수정본은 `--rl-config`로 지정한다.
초기 자세와 flap 파지 판정은 기존 S63 실험을 사용하고, 몸통·양팔·평면 base를
제어한다. A/T 실행·일시정지, B/R 초기화, X/C recenter, Y/H HUD는 동일하다.
데이터 저장이나 정책 학습은 수행하지 않는다. RL reward debug에는 에피소드 시간제한이
없으며, `pick`과 `pick_place` 모두 성공·실패 판정 또는 수동 일시정지/reset까지 계속
조작할 수 있다. 학습 환경의 에피소드 제한 시간은 해당 RL config를 따른다.

실물 Quest→motor command 지연까지 포함해 비교할 때만 예제의
`--arm-response responsive`를 `--arm-response real`로 바꾼다. `real`은 200ms 입력
queue를 사용하며 reset·일시정지·추적 중단 시 즉시 비운다. RL policy action이 실물의
VR 입력 경로를 거치지 않는다면 학습 환경에는 이 지연을 넣지 않는다.

- `pick`: 실제 flap 파지 + 초기 높이보다 6 cm 이상 lift + 기울기 허용치를
  0.5초 유지하면 carry로 넘어간다. 이 단계에서 전체 성공으로 종료하지 않는다.
- `carry`: 박스 전체 바닥 투영이 실제 Rack USD의 경계 밖으로 2.5 cm 이상
  빠져나와야 한다. 파지를 유지하며 컨베이어 목표 XY 15 cm 안에 도달하고,
  박스 바닥이 belt 상면 위에 있는 상태를 0.5초 유지하면 place로 넘어간다.
  선반보다 낮은 belt로 운반할 때 초기 선반 높이를 계속 유지하도록 요구하지 않는다.
- `place`: 전체 박스가 belt 안에 들어가고 높이·기울기가 맞으며, box Body와
  belt의 실제 접촉력 ≥0.2 N, 손 release, 선속도 <0.08 m/s 및 각속도
  <0.35 rad/s 조건을 0.5초 유지해야 전체 성공이다. 공중에 떠 있는 박스나
  계속 손으로 잡고 있는 박스는 성공으로 인정하지 않는다.

HUD에 현재 단계, 선반 인출 잔여 거리, 컨베이어 목표 거리, 지지 접촉력과
단계별 미충족 조건을 표시한다. 인출·운반·내려놓기 reward는 직전 스텝보다
남은 거리가 줄면 +, 늘면 −인 진행량이며 정지에는 0이다. 파지 손실·재획득 및
단계 전환에서는 진행량 기준을 다시 잡아 중간 이동을 소급 보상하지 않는다.
가중치는 `rl/managers/rewards.py`의 `FlapPickPlaceRewardsCfg`에서 조정한다.
`pick_place` 기본값은 robot-obstacle 접촉이 20 N을 넘으면 실패하며 collision penalty도 적용한다.
파지·팔·몸통 동작부터 분리해서 확인할 때는 `--no-rl-obstacle-collision`을 추가한다.
이 옵션은 obstacle collision 실패와 penalty만 끄며, **실제 PhysX 접촉, 장애물 force 센서와
HUD의 force 표시는 유지한다.** 따라서 물체를 관통시키는 옵션이 아니며 실물 안전 검증에도
사용하면 안 된다. 시작 로그와 HUD의 `Collision termination/penalty: OFF`로 적용 여부를 확인한다.
Base는 각 environment 중심에서 반경 1.5 m를 벗어나면 실패한다. 실제 적용되는 base 속도에는
항상 작은 비용을 주고, 파지 후 lift와 place 단계에는 추가 정지 비용을 준다. HUD의 `BASE` 줄에서
병진 속도, yaw 속도와 현재 중심 이격/한계를 확인할 수 있다.

## Dynamic base (박스를 든 채로 주행할 때)

dynamic base가 기본값이다. root를 floating으로 풀고 joystick 명령을 PD wrench로 추종하므로
로봇·gripper·박스가 같은 PhysX 해에서 함께 가속한다. 예전 kinematic base는 root pose를 매
physics step 덮어써서 gripper가 순간이동했고, 주행을 시작하면 물려 있던 박스가 미끄러졌다.
그 동작이 필요하면 `--no-dynamic-base`(또는 `KUAVO_DYNAMIC_BASE=0`)로 되돌린다. 이 선택은
teleop 수집, RL reward debug, RL 학습에서 동일하게 해석된다(`robots/base_drive.py`).

```bash
./quest_collector.sh collect \
  --robot-model s63 --gripper leju-twofinger \
  --controller-mapping absolute --absolute-orientation downward \
  --arm-response responsive \
  --rl-reward-debug 1 --rl-task pick_place \
  --no-rl-obstacle-collision --no-rack-rollers \
  --no-quest-camera-overlay --no-camera-preview \
  --no-wrist-cameras --no-head-camera
```

이때 world joint가 풀려 root가 floating이 되고(`fix_root_link=False`), joystick 명령을
적분한 x/y/yaw 목표를 root wrench로 추종한다. root state는 덮어쓰지 않는다.

수입된 바퀴는 실제 omni roller가 아니라 원기둥 collider다. floating root에서 이
접촉을 남기면 219 kg 로봇의 마찰원(바닥 마찰 0.8 이상)을 chassis wrench로 깨야 하므로
횡이동과 제자리 회전이 아예 걸리고 전진도 사실상 멈춘다. 그래서 floating spawn에서는
바퀴 지면 접촉을 제거하고, drive가 전체 무게(중심 오프셋 토크 포함)를 feedforward로
상쇄한 뒤 높이·기울기 PD로 차체를 떠받친다. 바퀴는 시각적으로만 회전한다.

게인은 `robots/base_drive_control.py`의 `FloatingBaseDriveCfg`에 있고, RL의
`PlanarDriveCfg.drive`와 teleop의 `TeleopBodyActionCfg.drive`가 같은 값을 쓴다. 추종이 무르면
`position_stiffness`/`velocity_damping`과 `max_linear_acceleration`을 올리고,
박스가 여전히 미끄러지면 `max_linear_acceleration`을 낮춰 가속을 완만하게 한다.
arms-only(reward debug `0`)는 base action 자체가 없어 고정 root를 유지한다. 기본값에서는
조용히 그대로 두고, `--dynamic-base`를 명시하면 충돌을 알린다. 바퀴가 없는 모델(S56)도 같다.

### 주행 중 상체 자세 유지 (측정값)

kinematic base는 root를 순간이동시키므로 관성 결합이 아예 없어 주행 중 관절 변화가
0.0°다. dynamic base는 실제로 관절에 하중이 걸리므로 편차가 생긴다. 전진·횡이동·회전·
대각 주행을 각각 2초씩 명령하며 목표 대비 최대 편차를 측정한 결과다.

| 관절군 | floating + 중력 feedforward만 | floating + `s63-body-id` (현재) |
| --- | --- | --- |
| 양팔 14축 | peak 0.26~0.57° | peak 0.05~0.30°, 종료 시 ≤0.05° |
| waist | peak 0.38~0.77° | peak 0.08~0.30°, 종료 시 ≤0.01° |
| 승강축(knee/leg) | peak 1.07~1.90° | peak 0.03~0.45°, 종료 시 ≤0.02° |
| head | 주행과 무관한 상시 1.18° 오프셋 | 동일 |

차이는 팔 추종 정확도가 아니라 torso 축 감쇠에서 나온다. 팔만 sine으로 움직였을 때의
추종 지연은 고정 root + ID가 peak 2.68°/rms 1.23°, floating + 중력만이 peak 2.44°/
rms 1.25°로 사실상 같았다. 반면 승강축은 ID의 가속도 task가 빠지면 감쇠비가 낮아져
(`docs/S63_GRAVITY_COMPENSATION.md` 참조) base 가속마다 1.9°까지 출렁였다.

그래서 floating root에서도 joint ID를 사용한다. generalized 배열의 joint block으로
관절 토크를 계산하고, planar drive는 base의 높이·기울기와 무게 지지를 독립적으로 맡는다.
목표 관절 가속도에서 계산한 root 반작용은 접촉·토크 제한 아래 실제 가속도와 달라져
base를 가진할 수 있으므로 feedforward하지 않는다. dynamic base는 actuator gain을 학습
scene과 동일하게 유지한다.

head의 1.18°는 정지 상태와 kinematic base에서도 동일하게 나타나는 기존 target
오프셋이므로 base 움직임과 무관하다.

### 자세 제어 축과 관성 (2026-09-19 수정)

섀시 자세 오차는 world frame rotation vector로 계산한다. 초기 구현은 Euler XYZ의
roll/pitch를 읽어 world x/y 토크로 보정했는데, 그 각도는 몸체 축 기준이라 base를
90° 돌리면 roll 보정이 pitch 축에 걸려 감쇠 대신 가진이 됐다. 회전 후 팔을 휘두르면
기울기가 22.7°→30.2°로 발산했고, 수정 후 같은 동작은 0.04°에서 0.00°로 잦아든다.

기울기 관성도 평행축 항을 포함해 매 step 링크 위치에서 계산한다. 이전에는 로컬 관성
합과 `m·r²` 하한만 써서 tilt 축이 19.7 kg·m²로 잡혔지만 실제 값은 76 kg·m² 수준이다.
팔을 뻗으면 관성이 늘어난 만큼 토크도 함께 커진다. 팔이 랙에 닿는 순간처럼 접촉이
생기면 `max_tilt_acceleration` 상한(10 rad/s²)에 잠깐 걸리지만 곧바로 회복한다.

base의 수평·수직 이동력은 root body COM에 적용되므로 전체 무게중심이 root보다 위에 있으면
그 자체로 pitch/roll moment를 만든다. 특히 torso를 높이면 레버암이 커진다. drive는
현재 whole-body CoM까지의 레버암과 이동력의 외적으로 같은 step의 보상 couple을 더한다.
이 항은 PD gain을 높이지 않고 높이 변화와 주행 때 생기는 불필요한 pitch moment를 직접
보상하며, 실제 관절 반작용은 base 자세 PD가 감쇠한다.

RL scene은 `minimal_rl_v2`이며 local cuboid로 만든 정지 belt, 측면 프레임 및
4개 다리를 갖는다. 물리 상판은 2.55×0.68 m, 두께 3 cm이고 기존 높이를 유지한다.
위치는 `configs/workcell_layout.json`의 `conveyor` anchor를 공유한다.
인출·놓기 검사에서는 belt 이동과 버튼 누르기를 요구하지 않는다.
Scene profile이 바뀌었으므로 이전 checkpoint/reset bank와의 호환을 가정하지 않는다.

RL reward debug는 일반 Teleop의 factory 배경을 복제하지 않고 `minimal_rl_v2` workcell을
생성한다. 그래도 rack의 위치·회전·scale은 같은 `configs/workcell_layout.json`을 사용한다.
기본 SmallBox는 `configs/rack_box_poses_small_middle.json`의 고정 중간 선반 pose를 사용한다.
자동 배치 위치를 명시적으로 시험하려면 다음처럼 기본 캡처 pose를 끈다.

```bash
./quest_collector.sh collect \
  --rl-reward-debug 1 --rl-task pick_place \
  --rack-boxes '2:small' --ignore-captured-box-poses \
  --no-rack-rollers
```

## 실행

저장소 루트에서 기존과 같은 순서로 실행한다. Runtime/web이 이미 실행 중이면
중복 실행하지 않는다. 일반 Isaac 수집기와 이 검사 모드를 동시에 실행하지 않는다.

```bash
# 터미널 1
./quest_collector.sh runtime
# 터미널 2
./quest_collector.sh web
# Quest에서 기존 HTTPS 페이지 → CONNECT 후, 터미널 3
./quest_collector.sh collect --rl-reward-debug
```

`--rl-reward-debug`는 선택적 정수 값을 받는다. 값을 생략하거나 `0`이면 로드한
실험 설정의 오른팔 전용 action space를 유지해 base·torso·head와 왼팔/왼손을
물리적으로 고정한다. `1`을 주면 `all-joints` action space를 사용해 base, torso,
head와 양팔/양손을 모두 해제한다. 성공 조건인 `grasp_hand="right"` 한 손 파지
판정 자체는 바뀌지 않는다.

controller mapping은 `absolute`, `scaled`, `relative`를 사용할 수 있다. `relative`를
선택하면 일반 VR teleop과 같은 wrist delta 입력으로 로봇을 조작하고, 이 scene의
reward·contact·grasp/carry/place probe는 조작에 개입하지 않고 관찰값만 계산한다.

`2`는 `1`의 absolute/downward/responsive, 전체 관절, binary `0=open/1=close`
및 package 50 N gripper, 카메라 OFF
기본값을 이어받아 격리된 `rl/multi_box` v2 scene을 연다. 시작/reset마다 rack
(X/Y ±10 cm, yaw ±15°)과 conveyor pose 및 1~12개 box 배치를 다시 뽑는다.
두 번째 shelf에는 small/medium,
세 번째 shelf에는 small box만 생긴다. roller rack이 기본이며 필요할 때만
`--no-rack-rollers`로 plain rack을 선택한다. mode 2 HUD는 활성 box와
randomization 결과 외에 선택한 box의 v2 pose shadow reward를 표시한다. reset 때
양 TCP에서 가장 가까운 활성 box를 target으로 잠그며 PC J/L로 target을 바꿀 수 있다.
PC 1/2/3은 각각 grasp/carry/place dense reward를 검사한다. 이 phase 선택은 표시와
로그에만 적용되고 task state나 로봇 제어를 변경하지 않는다.

현재 shadow 값은 실제 TCP와 flap rigid-body 형상, box/rack/conveyor pose, box velocity로
계산한다. 네 손가락과 18개 physical box의 두 target flap 사이에서 PhysX가 이미 계산한
filtered contact도 읽는다. HUD/JSONL의 `CONTACT RAW`에는 현재 target box에 대한 손가락별
접촉력, 실제 flap bounds 안의 접촉인지와 두 jaw가 flap을 사이에 두는지를 표시한다.
`GRASP PROBE`는 jaw마다 최소 5 N, 두 jaw의 유효 접촉, 양손의 서로 다른 flap 파지,
손–박스 상대자세 변화 10 mm/10° 이내, 초기 위치보다 8 mm 상승하면서 경사진 선반과
8 mm 이상 간격을 0.25초 유지했는지를 읽기 전용으로 표시한다. 패드 변경 이후
5 N 기준의 재확인이 필요하다.
`CARRY PROBE`는 파지 성공 이후 양손 opposing-flap 접촉 유지, 중력 기준 박스 기울기
20° 이내, 박스 footprint의 벨트 안쪽 배치, 다른 박스와의 비중첩, 벨트 위 5–15 cm
높이를 표시한다. `PLACE PROBE`는 carry 성공
이후 실제 Box Body–belt 지지력 0.2 N 이상, 양쪽 physical grasp 해제,
긴 변 평행 10° 이내, 박스 속도 5 cm/s·0.2 rad/s 이하를 0.5초 유지하는지를 표시한다.
다른 박스와의 중첩은 placement state가 연결되기 전까지 벨트 근처의 모든 활성
박스를 보수적으로 장애물로 취급한다.

VR control이나 phase는 probe가 바꾸지 않는다. 대신 weight 검증용 sidecar에는
grasp/carry/place 세 phase를 매 step 동시에 계산한다. 승인된 probe의 pinch, skill
success, support, release와 grasp-loss 판정을 one-step event reward로 연결하며,
base/action/joint-limit 비용과 학습 환경의 aggregate obstacle, box, workspace guard도
같이 기록한다. 양손 pinch를 잃었다가 다시 잡는 행동으로 보상이나 terminal penalty를
반복 발생시키지 않도록 event는 scene reset 또는 target 변경 전까지 한 번만 지급한다.
30 Hz에서 수 초 이상 걸리는 skill의 성공 credit을 유지하도록 v2 potential과
SAC/PPO discount는 모두 0.999를 사용한다.
JSONL의 최상위 `phase`, `weighted_terms`, `step_reward`는 HUD에서 선택한 phase이고,
`weighted_terms_by_phase`, `step_reward_by_phase`, `potentials_by_phase`,
`raw_by_phase`에 세 phase 전체 값이 들어간다. 이 값은 weight 비율의 사전 검증용이며,
짧은 학습에서 reward 분포와 성공률을 재확인한 뒤 장기 학습을 시작한다.

매 control step의 raw SI 값, normalized potential, weighted term과 누적 return을 별도
JSONL에 저장하려면 존재하지 않는 새 경로를 지정한다. 데이터셋에는 섞이지 않는다.
각 행의 `raw.trial_index`는 시작 시 0이며 B/R reset마다 증가한다.
`raw.target_logical_id`와 `raw.active_box_count`로 어떤 장면과 박스를 검사했는지
구분할 수 있다. 누적 return은 reset 때 다시 0에서 시작한다. A/T 일시정지·재개와
X/C recenter는 물리 장면을 reset하지 않으므로 probe의 이전 성공 이력을 유지한다.
대상 박스 변경(J/L)과 장면 reset(B/R)은 그 이력을 초기화한다.

```bash
./quest_collector.sh collect --rl-reward-debug 2 \
  --rl-shadow-box-count 1 \
  --rl-shadow-log artifacts/multi_box_grasp_probe_01.jsonl
```

접촉력과 파지 미끄럼을 보정할 때는 target이 실제 조작한 박스와 달라지는 것을 막기 위해
`--rl-shadow-box-count 1`로 한 개만 생성한다. 이 옵션은 mode 2 진단 scene에만 적용되며
학습용 full task의 1–12개 분포를 변경하지 않는다.

```bash
./quest_collector.sh collect --rl-reward-debug 1   # 전체 관절 풀기
./quest_collector.sh collect --rl-reward-debug 2   # multi-box v2 VR teleop 검사
./quest_collector.sh collect --rl-reward-debug 0   # 기존과 동일: 활성 팔만
./quest_collector.sh collect --rl-reward-debug     # 값 생략 시 0과 동일
```

V2 grasp SAC에 쓸 Quest 시연 전이를 기록하려면 mode 2에 **별도 파일 옵션**을 붙인다.
기존 수집(옵션 없음)과 mode 1/2 보상 검사(파일 옵션 없음)는 그대로 동작한다.

```bash
CUDA_VISIBLE_DEVICES=0 ./quest_collector.sh collect \
  --robot-model s63 --gripper leju-twofinger --rl-reward-debug 2 \
  --device cuda:0 --no-rl-demo-self-collision \
  --rl-demo-dataset datasets/v2_grasp_quest_001.hdf5
```

예시는 self-collision을 끈 SAC 실행에 맞췄다. 켠 실행의 시연을 모을 때에는
`--no-rl-demo-self-collision`을 빼면 된다. 기본값은 V2 학습 설정과 같이 켜짐이다.

이 경로는 mode 2의 전체 장면 shadow reward 대신 **실제
`MultiBoxGraspAssemblyEnvCfg` staged-grasp 환경**을 실행한다. 따라서 보상, 종료,
관측, 정규화 전 25차원 관절 action은 V2 SAC와 같은 코드에서 나온다. `A/T`로
조작/일시정지, `B/R`로 현재 시도 종료·새 장면 reset, `X/C`로 recenter한다.
성공·안전 위반·30초 timeout 때에는 마지막 terminal 관측을 저장하고 자동으로
일시정지한다. 다음 시도는 `A/T`로 시작할 수 있다. 초기 박스 settling 단계의
zero-action frame은 데이터에 넣지 않는다. `--max-episodes`로 종료할 시도 수를
지정할 수 있다.
reset 검증 실패(`invalid_reset`)가 발생하면 학습기와 같이 해당 전이는 제외하고
시도를 중단한다.
기존 reward-debug preset의 `--no-rl-obstacle-collision`은 이 수집 경로에
적용되지 않는다. V2 학습의 장애물 충돌 판정을 그대로 사용한다.

### 안전 종료(unsafe) 원인 확인

안전 위반으로 끝나면 걸린 판정과 측정값, 그리고 접촉한 링크를 콘솔에 함께 출력한다.

```
[V2 DEMO] episode_000000: unsafe (robot_rack_collision), success=False
[V2 DEMO]   rack 41.2/10.0 N | obstacle 0.0/5.0 N | base 0.22/1.50 m | self-collision off | box z 1.21 m | lift 0.00/0.50 m | box speed 0.03/10.0 m/s, 0.10/100.0 rad/s
[V2 DEMO]   rack zarm_r4_link 41.2 N at (1.05, -0.21, 1.48) m
```

판정은 `robot_rack_collision`(랙·롤러 접촉 10 N 초과), `obstacle_collision`(랙을 제외한
박스·벨트·바닥 접촉 5 N 초과), `self_collision`, `workspace_limit`(base 중심에서 1.5 m),
`box_drop`, `box_lift_limit`(0.5 m), `box_speed_limit` 일곱 가지다. 표시값은 termination
manager가 그 스텝에 이미 계산한 스냅숏을 읽은 것이며 다시 측정하지 않는다. HDF5의
`end_reason`은 기존과 같이 `unsafe`로 저장한다.

충돌 판정 대상은 손가락 4개를 제외한 `base_link`, `waist_yaw_link`,
`zarm_l2/l4/l7_link`, `zarm_r2/r4/r7_link`, 좌우 `twofinger_base` 10개 링크다. 손가락과
박스의 접촉만 파지 신호이므로, **손바닥(`twofinger_base`)이나 팔뚝이 박스를 5 N 넘게
누르면 장애물 충돌로 종료**된다. 랙 접촉은 롤러 포함 10 N이 기준이다.

접촉 중인 링크 위치에는 기본으로 구를 표시한다. 빨강은 랙·롤러, 주황은 그 외
(박스·벨트·바닥)이고 1 N부터 나타난다. 종료를 일으킨 접촉 표시는 `A/T`로 다음 시도를
시작하거나 `B/R`로 reset할 때까지 남는다. 콘솔에도 1초에 한 번
`[V2 CONTACT] rack zarm_r4_link 6.2 N at (...) m`을 출력한다. 구는 링크 원점에 그리므로
접촉 패치의 정확한 위치는 아니며, 물리·보상·관측에는 들어가지 않는다. 표시를 끄려면
`--no-rl-demo-contact-markers`를 추가한다.

실패·성공으로 끝난 프레임은 Isaac의 자동 reset 뒤에 **종료 직전 로봇·박스 자세로 되돌려
정지 상태로 유지**한다. 그래서 어디서 부딪혔는지 그 장면 그대로 둘러볼 수 있다. 이때
물리는 진행하지 않으며 표시 전용 복원이다. `A/T`를 누르면 실제 `env.reset()`으로 새 장면을
만들고 다음 시도를 시작하고, `B/R`도 즉시 새 장면으로 넘어간다. 기록된 전이는 이미 종료
시점에 저장되므로 이 복원은 데이터에 영향을 주지 않는다. 바로 reset되던 이전 동작이
필요하면 `--no-rl-demo-hold-terminal-frame`을 추가한다.

## 성공 데모만 추출해 공유하기

`scripts/rl/export_demo_subset.py`는 기록된 파일에서 **완료된 성공 에피소드만** 새 파일로
복사한다. 값은 수정하지 않고 gzip으로만 압축하며, 기존 파일을 덮어쓰지 않는다.

```bash
python3 scripts/rl/export_demo_subset.py \
  datasets/v2_grasp_quest_005.hdf5 --output datasets/v2_grasp_success.hdf5
```

여러 파일을 함께 주면 병합하되, 관측·행동·보상 계약(`action_dim`, 관측 차원,
`control_dt`, gripper force, roller, `multi_box`/`task` 설정)이 다르면 거부한다.
`controller_mapping`이나 seed처럼 사람이 조작한 방식만 다르면 병합하고 manifest의 해당
키를 `"mixed"`로 표시하며, 에피소드마다 `source_file`/`source_episode`를 남긴다.
`--all-episodes`는 실패 에피소드까지, `--allow-incomplete`는 중단된 기록까지 포함한다.

예시 결과물은 [examples/demos](../examples/demos/README.md)에 올려 두었다.


파일은 독점 생성하며 기존 경로를 덮어쓰지 않는다. 루트 `manifest_json`에는
로봇, 그리퍼, 행동 항목 순서/차원, 관측 차원, 제어 주기, 설정을 담는다.
`episodes/episode_XXXXXX/transitions`에는 action 직전 `actor_obs`와
`critic_obs`, 실제 적용한 `action`, `reward`, terminal reset 이전의
`next_actor_obs`/`next_critic_obs`, `terminated`/`truncated`, 성공·안전 종료
플래그와 simulation time을 저장한다. `critic_obs`는 SAC replay와 같이
`policy + privileged` 결합이다. 사람이 B/R로 중단한 에피소드 및 프로세스 종료
시 미완료 에피소드는 실패/미완료로 표시한다. **이 파일의 수집은 구현되었지만,
기존 SAC 학습기에 데모를 자동 주입하는 loader는 아직 없다.** 먼저 데이터 품질과
동일 설정 재생을 확인해야 한다. mode 1은 legacy task 계약이라 V2 SAC 데이터셋
옵션을 받지 않는다. `--rl-shadow-box-count`/`--rl-shadow-log`는 전체 장면
진단 전용이므로 함께 지정할 수 없다.

전체 관절 모드에서는 왼쪽 컨트롤러의 위치·회전도 왼팔 IK 목표로 쓰이므로, 실행 중
추적 유효성 검사(`tracked`)도 왼쪽 컨트롤러까지 함께 요구한다. 콘솔의
`[RL REWARD] control=..., action_space=...`로 실제 적용된 값을 확인할 수 있다.

좌우 wrist 영상 패널 옵션도 유지한다. reward 텍스트와 stereo 장면만으로
가볍게 확인하려면 `--no-quest-camera-overlay --no-camera-preview`를 추가한다.
이 경우 RGB 카메라 센서를 생성하지 않는다.

성능을 위해 PC 전체 viewport, collider 표시, grasp marker는 기본 OFF다.
필요한 검사에서만 각각 `--desktop-render`, `--rl-collision-view`,
`--rl-grasp-markers`를 추가한다. Reward HUD는 기본 ON이며 최대 5 Hz로만
갱신한다. HUD 비용까지 제외해 비교하려면 `--no-rl-reward-hud`를 사용한다.
collider/marker를 끄면 해당 시각화 geometry 생성과 표시용 갱신만 생략한다.
Reward가 사용하는 실제 접촉·충돌 판정은 환경 재현을 위해 계속 계산한다.
콘솔의 `[PERF]`는 XR loop Hz와 실제 physics/control Hz를 5초마다 구분해 출력한다.

롤러를 끈 기본 RL Debug는 한 환경의 실제 조작 속도를 우선하는 profile을 사용한다.
물리는 120 Hz, 제어·aggregate collision·reward는 30 Hz로 계산하고 robot/box solver는
RL 설정을 유지한다. 30 Hz 물리와 8/2 solver로 낮추던 이전 경로는 관절 범위 이탈과
전신 자세 불안정을 줄이기 위해 제거했다. 정책을 실행하지 않으므로 policy observation과 학습용 success snapshot
recorder를 만들지 않으며, reward 뒤 observation을 준비하기 위한 중복 측정도 생략한다.
충돌 판정은 손가락 이외의 robot contact 합력과 손가락의 비정상 residual contact를
사용한다. 물리 contact 자체를 끄는 설정은 아니다.

```bash
# 롤러 OFF 실시간 검사: 이 PC에서 권장
./quest_collector.sh collect --rl-reward-debug 1 \
  --no-rack-rollers --device cpu

# 학습 환경의 120 Hz filtered contact와 정확히 비교할 때
./quest_collector.sh collect --rl-reward-debug 1 \
  --no-rack-rollers --device cuda:0 --rl-obstacle-contact-hz 120
```

이 단일 환경에서는 CPU PhysX가 CUDA PhysX보다 빨랐다. `--device cuda:0`을 명시하면
롤러 OFF 실시간 profile 자체는 적용되며 실제 속도는 `[PERF]`로 확인한다.
`--rl-obstacle-contact-hz`를 명시하면 aggregate 접촉 대신
per-body filtered obstacle sensor를 사용한다. 선택값은 `15`, `30`, `60`, `120` Hz다.
충돌 임계값이나 학습 결과를 최종 비교할 때는 120 Hz를 사용한다.

롤러를 켜면 546개 roller rigid body의 물리가 필요하므로 기존 120 Hz 물리와 filtered
sensor 경로를 유지한다. 이때 obstacle report 기본값은 30 Hz다. Reward manager 자체를
늦추면 성공 hold 시간과 progress reward가 지연되는 데 비해 이 PC에서 약 1~2%만 줄어
별도 decimation을 적용하지 않는다.

기존에 `collect_quest_teleop.sh`를 직접 사용했다면 기존 명령 끝에
`--rl-reward-debug`를 추가해도 된다. `XR_RUNTIME_JSON` 등 기존 연결 설정은 필요하다.
체크포인트는 필요 없다. 기존 학습을 중단하거나 재시작하지 않지만, GPU 자원이
부족한 PC에서는 학습과 XR을 동시에 실행하지 않는 것이 좋다.

## 조작

기본 모델은 S63 + `leju-twofinger`이고, 몸통 4관절·양팔 14관절에 물리 스텝마다 공통
중력 보상을 적용한다. 이 모델은 임시로 높인 몸통 PD 대신 `configs/s63_servo.json`의
설정을 사용한다. 초기 자세는 S63용 `s63_leju_ready_01`이다. S200062도 같은 중력
보상 경로를 사용하고 기존 `quest_ready_02`를 유지한다. S56 + `s56_twofinger`는
다리까지 보상하며 별도 `s56_twofinger_ready_01`을 사용한다. 모든 보상 모델에서
8000/200 등의 추가 몸통 PD 보강은 적용하지 않는다.
물리적으로 몸통을 고정하는 오른팔 전용(`0`)에서는 잠긴 관절에 중력 보상을 추가하지 않는다.
[S63 중력 보상·PD 설정](S63_GRAVITY_COMPENSATION.md)을 참고한다.
스틱이 중립이면 마지막 몸통 목표 각도를 PD로 유지한다. 초기 대기·일시정지·재개·시점
보정 때도 RL action manager의 목표 각도를 사용하며, 중력으로 처진 실제 각도를 새 목표로
받아들이지 않는다. 스틱을 놓은 직후에는 가속도 제한에 따른 짧은 감속 구간이 있다.
`1` 모드는 관절 잠금이 아니므로 하중에 따른 작은 자세 오차는 남을 수 있다.
실제 팔 각도가 관절 범위를 0.02 rad 넘게 벗어나면 앱을 종료하지 않고 물리·제어를 정지한다.
`[RL CONTROL]`에 관절 이름·실제 각도·범위를 출력하고 `ROBOT JOINT LIMIT` 패널을 표시한다.
이때는 A 대신 B/PC R로 reset해야 한다. URDF/USD 모델 불일치는 여전히 오류로 처리한다.

| 입력 | 검사 모드 동작 |
|---|---|
| X / PC C | 시점 보정 후 일시정지 |
| A / PC T | 물리·팔 추종 시작/일시정지, 손 위치·회전 기준 재설정 |
| 왼쪽 joystick | `1`/`2` 모드에서 base 전후·좌우 이동 |
| 오른쪽 joystick 좌우 / 상하 | `1`/`2` 모드에서 base 회전 / torso 승강 |
| 오른쪽 아래 그립(squeeze) + 오른쪽 joystick 상하 | `1`/`2` 모드에서 몸통 전후 이동: 위=앞, 아래=뒤 |
| 오른쪽 아래 그립(squeeze) + 오른쪽 joystick 좌우 | `1`/`2` 모드에서 waist yaw 회전 |
| 오른쪽 검지 트리거 | 오른쪽 gripper 닫기; 놓으면 열기. 기본 오른손 실험은 왼쪽 입력 무시 |
| B / PC R | 모델별 초기 자세로 초기화하고 정지 (**녹화 버튼 아님**) |
| Y / PC H | reward 패널 표시/숨김 |
| PC J / L | mode 2 shadow target을 이전/다음 활성 box로 변경 |
| PC 1 / 2 / 3 | mode 2 shadow phase를 grasp / carry / place로 변경 |

아래 그립은 중지 `squeeze`, gripper 개폐는 위 검지 trigger다. 오른쪽 squeeze를
놓으면 오른쪽 joystick은 base 회전/torso 승강으로 돌아가고 몸통 전후·waist yaw
목표는 유지된다. 왼쪽 joystick의 base 전후·좌우 이동은 modifier 중에도 유지된다.

정면을 보고 X → 양 컨트롤러를 편한 자세에 두고 A → 손을 움직여 검사한다.
도달 범위가 부족하면 A로 멈추고 손을 편한 위치로 옮긴 뒤 A로 다시 시작한다.
일시정지 동안 물리·task 시간은 멈추고 VR 화면은 계속 렌더링된다.
추적이 끊겨도 정지하며, 복구 후 A를 눌러야 재개한다.

성공·실패 때 마지막 reward를 고정 표시한다. RL 내부 자동 reset은 유지하므로
장면은 초기 상태로 돌아가지만 패널은 종료 직전 값이다. B로 새 시도를 준비한다.
초기 안정화 동안은 `SETTLING`으로 표시하고 손 입력을 적용하지 않는다.

## VR 표시값

### 실제 collider·안쪽 면 후보·접촉점·힘 표시

`--rl-collision-view`를 명시하면 실제 collider와 접촉 표시를 생성한다. 기본값은
성능을 위해 OFF다. 기존의 수동 offset 구와 구분되는 표시이며, offset을 입력하지
않아도 실제 형상을 읽는다. 관련 코드는
`rl/debug/collision_overlay.py`, CPU 기하 계산은 `collision_geometry.py`다.

```bash
# collider와 기준점/목표점을 함께 검사할 때만 둘 다 명시한다.
./quest_collector.sh collect --rl-reward-debug \
  --rl-collision-view --rl-grasp-markers
```

| 색/형태 | 실제 표시 데이터 |
|---|---|
| 청록 윤곽선 | 활성 오른손 두 손가락 collider. mesh는 PhysX cooking convex 결과 |
| 주황 윤곽선 | 두 후보 flap collider. USD Cube는 실제 primitive 크기, mesh는 PhysX cooking 결과 |
| 초록 굵은 면 테두리 + 구 | 상대 손가락 방향을 향한 collider polygon 후보와 그 면적 중심 |
| 보라 상자 윤곽선 | 현재 RL 파지 판정의 bounds + contact margin. top_band 모드면 해당 상단 영역 |
| 빨간 구 + 화살표 | PhysX 센서의 평균 접촉 위치와 해당 손가락–flap 쌍의 법선 접촉 합력 |

**초록 면은 자동 선정된 안쪽 면 후보이며, 실제 패드라는 의미가 확정된 것은 아니다.**
상대 손가락의 collider 중심 방향과 법선이 60° 이내로 정렬된 polygon 중
`면적 × cos(각도)^4`가 가장 큰 것을 선택한다. 오른손 두 손가락 각각 독립적으로 선정한다.
중심은 삼각분할 면적 가중 중심이다. 경계선은 z-fighting을 줄이기 위해 법선 방향으로
0.5mm만 띄워 그리며, 중심 구는 실제 면 위에 둔다. 면 후보는 손가락 자세에 따라 바뀔 수 있다.
이 후보를 보상·파지 판정에 자동 적용하지 않는다.

PhysX cooking 요청은 실행 중인 stage에서 활성 CollisionAPI와 approximation 설정을 읽어
비동기로 수행한다. 원본 visual mesh나 전체 visual bounds를 실제 collider라고 대체하지 않는다.
collider 로컬 형상·내부 scale을 body 좌표계로 변환해 저장하고, 표시 시 live articulation
위치/회전을 적용한다. 지원하지 않는 collider 형식이나 cooking 실패는 경로와 원인을 로그에
남기고 제외한다. `COLLIDERS ready / pending / unavailable`은 그 읽기 상태다.
형상/scale 자체를 편집하면 검사 프로세스를 재시작해야 한다. 표시 중 비활성화된 collider는 숨긴다.

접촉 위치는 **전체 접촉 패치가 아니라 센서의 평균점**이다. 화살표는 평균점에 해당 쌍의
법선 접촉 합력 벡터를 붙여 표시한 것이며 개별 접촉점의 힘이 아니다. 마찰의 접선력·토크는
포함하지 않는다. 힘이 finger에 작용하는 방향을
보여주며 길이는 1cm/N, 최대 12cm로 제한한다. 실제 N 값은 HUD와 기존 진단 로그에서 읽는다.
접촉점이 NaN이거나 힘이 없으면 해당 구/화살표를 숨긴다. 기준점 offset은 사용하지 않는다.

- PC Isaac 창에서 **J**: collision 표시 전체 ON/OFF. **G**: 기존 기준점·목표점 표시.
- **Y**는 reward 패널만 제어한다. HMD 안의 3D 형상 표시는 J/G와 독립적이다.
- `--no-rl-collision-view`로 기능을 비활성화할 수 있다. 양 기능을 모두 꺼도 보상 판정은 같다.
- 3D 형상은 일반 RTX scene geometry이므로 VR 시점에서도 렌더링되는 경로다. 다른 물체를
  투시하는 X-ray는 아니므로 가리면 시점을 옮긴다. 초록 후보가 실제 안쪽 패드인지 먼저 확인한다.
- 자동 reset 직후에는 이전 episode의 접촉점을 새 자세에 그리지 않도록 숨긴다. collider는
  현재 reset 장면을 따른다. 일시정지 중에는 마지막 유효 접촉 상태를 보여준다.
- 표시 갱신은 약 10Hz이며 짧은 접촉을 모두 기록하지 않는다. 접촉 이벤트 저장기가 아니다.

이 기능은 보상·관측·collider·관절·마찰을 변경하지 않는다. 시각화 prim에는 물리를 적용하지
않는다. rendering/cooking 오버헤드는 검사 모드에만 추가된다. CPU 기하 테스트와 문법 확인만
수행했으며, 실제 cooking 완료·VR 가시성은 사용자가 실행 후 확인해야 한다.

### 양팔 손가락 기준점을 드래그해서 맞추기

현재 S200062는 [공통 endeffector_center](ENDEFFECTOR_CENTER.md)가 기본 reward/IK/VR 표시 기준입니다.
flap-pick `reaching`은 파지 전 직전 스텝보다 접근하면 +, 후퇴하면 -, 정지하면 0입니다.
파지 획득/유지 중에는 0이고, 파지를 잃은 첫 스텝에는 현재 거리로 기준만 다시 잡습니다.
`Reach signed progress`, `Reach baseline score`, `Reach tracking`을 함께 확인하세요.
`Reaching last 0.5s | episode`는 모든 제어 스텝의 실제 weighted reaching 보상 합계입니다.
순간값이 0이어도 최근 합계나 누적에 짧은 접근 보상이 남습니다. 양/음은 서로 상쇄될 수 있습니다.
최근 구간은 시뮬레이션 시간 기준이며 pause/terminal에서 보존하고 B/reset으로 초기화합니다.
정렬·lift도 개선/상승에는 +, 악화/하강에는 -인 변화량 보상이며, 파지는 에피소드당
한 번만 보상합니다. stable_grasp 지속 보상은 비활성화했습니다. 기존 checkpoint는
reward/관측 의미·차원이 달라 새 학습이 필요합니다.
아래의 편집 모드는 저장 후 재시작할 때 반영하며, 실행 중인 RL 정의를 즉시 바꾸지 않습니다.

**마우스로 위치를 맞추는 작업은 [데스크톱 전용 보정 도구](GRASP_DESKTOP_CALIBRATION.md)를
사용하세요.** 아래는 VR 내부 표시 옵션이며, XR 화면은 일반 데스크톱 편집에 적합하지 않습니다.
데스크톱 도구는 헤드셋 없이 실행하며 동일한 네 기준점 파일을 저장합니다.

저장소 루트에서 실행합니다. 이 모드는 기존 기준점/목표점 마커 대신
양팔의 손가락 기준점 **4개**를 표시합니다. 시뮬레이션 물리나 RL 보상은 변경하지 않습니다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-grasp-calibration \
  --desktop-render --no-rl-collision-view
```

1. Quest A 또는 데스크톱 T로 **일시정지**합니다. Isaac Sim의 Stop 대신 수집기의 pause를 사용합니다.
2. Isaac Sim Stage에서 아래 경로의 **`tip` Xform**을 선택합니다.
   부모 손가락 프레임이나 자식 `marker`, 실제 로봇/collision prim은 이동하지 마세요.

   ```text
   /Visuals/GraspCalibration/
     l_f_finger/tip    왼쪽 f, 주황색
     l_b_finger/tip    왼쪽 b, 분홍색
     r_f_finger/tip    오른쪽 f, 청록색
     r_b_finger/tip    오른쪽 b, 파란색
   ```

3. Viewport의 이동 도구로 각 점을 원하는 손가락 안쪽 접촉 위치에 맞춥니다.
   로컬/월드 이동 모드 모두 사용할 수 있습니다. 회전·스케일은 보정 대상이 아닙니다.
   점이 손가락 내부에 가려져 있으면 Stage에서 `tip`을 선택한 뒤 이동하세요.
4. 입력 필드 편집을 끝내고 Viewport에 키보드 포커스를 둔 뒤 **K**를 누릅니다.
   터미널의 `[GRASP CALIBRATION] Saved ...`로 저장 여부를 확인하세요.
   실행 중에는 저장이 거부되므로 먼저 A/T로 정지해야 합니다.
5. A/T로 재생하여 점이 손가락을 따라가는지 확인합니다. 현재 오른팔 전용 RL 설정에서는
   왼손이 고정되어 있으며, 이 보정 모드가 왼팔 제어를 활성화하지는 않습니다.

네 좌표는 작업 디렉터리 기준 `configs/grasp_reference_points.json`에 각 손가락 링크의
로컬 XYZ(m)로 저장됩니다. K를 다시 누르면 해당 파일을 갱신합니다.
다음에 같은 모드로 실행하면 자동으로 불러옵니다. 별도 파일은
`--rl-grasp-calibration-file configs/my_grasp_points.json`으로 선택합니다.
저장 파일의 로봇 모델이 다르면 불러오기를 거부합니다.
처음 실행하여 저장 파일이 없으면 네 점은 **손가락 링크 원점**에 표시됩니다.
저장 없이 종료한 편집은 사라집니다. USD 씬 저장은 필요하지 않습니다.

G는 네 점 표시를 토글합니다. HUD에도 현재 네 로컬 좌표가 표시됩니다.
S200062의 공통 보정 파일은 다음 실행부터 TCP/reaching/파지 기하에도 적용됩니다.
현재 실행 중인 보상 정의는 편집 중 바뀌지 않습니다. 저장 후 재시작하세요.

### 오른손 기준점과 목표점의 3D 표시

두 손가락 기준점만 먼저 확인하려면 다음 명령을 사용한다. flap 목표점과 collision overlay를
숨기고 청록/파란 구만 남긴다. HUD에는 각 local offset(m)과 두 기준점 간격(mm)이 표시된다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-grasp-markers \
  --no-rl-endeffector-centers --no-rl-grasp-targets --no-rl-collision-view \
  --rl-grasp-finger-offsets 0 0 0 0 0 0
```

S200062에서는 모두 0인 CLI 값 대신 저장된 보정값을 기본으로 불러온다.
보정이 없는 다른 모델에서는 link 원점이다. 로컬 좌표는 손가락 link와 함께
움직이므로, 올바른 점이면 팔을 움직이고 그리퍼를 열고 닫아도 동일한 패드 위치에 붙어 있어야 한다.

`--rl-reward-debug`에서는 기본적으로 4개 marker를 월드 좌표계에 표시한다.
물리·보상·관측에는 들어가지 않으며 일반 수집기와 학습 runner는 생성하지 않는다.

| 표시 | 의미 |
|---|---|
| 청록 구 | `r_f_finger` 기준점 |
| 파란 구 | `r_b_finger` 기준점 |
| 주황 작은 큐브 | f 손가락의 flap 목표점 |
| 분홍 작은 큐브 | b 손가락의 반대 면 목표점 |

**기본 offset=0은 link 원점이지, 검증된 fingertip/패드 중심이 아니다.** 두 구가 실제
안쪽 접촉 패드의 끝에 놓이는지 먼저 확인한다. 손가락별 로컬 offset은 다음 옵션으로 지정한다.
단위 m, 순서는 f의 x/y/z 다음 b의 x/y/z다. 아래 0을 보정값으로 바꾸고 검사 프로세스를 재시작한다.

```bash
./quest_collector.sh collect --rl-reward-debug \
  --no-rl-endeffector-centers \
  --rl-grasp-finger-offsets 0 0 0 0 0 0 \
  --rl-grasp-marker-flap flap_right
```

각 손가락의 로컬 축이 다를 수 있다. CLI에서 지정한 비영점 offset은 표시 전용이며
공통 보정 파일이나 reward TCP를 바꾸지 않는다. 목표점은 두 기준점 중점을 flap 면 범위 안에 clamp한
같은 접선 위치의 반대 면 한 쌍이다. 손가락/면 배정은 두 거리 중 큰 값을 작게 하는 쪽으로 고른다.
`auto`(기본)는 현재 잡고 있는 flap을 유지하고, 미파지 시 쌍 거리 비용이 작은 후보를 고른다.
`flap_right`/`flap_left`는 표시 후보만 고정한다. **새 reach 설계 미리보기이며 기존 reaching
reward 목표점은 아니다.** HUD의 `PREVIEW`에 후보와 두 기준점–목표점 거리를 표시한다.

- X로 시점을 정렬하고 A로 움직인다. 다시 A로 멈춘 뒤 정지한 상태에서 둘러볼 수 있다.
- PC Isaac 창에 초점을 두고 G로 marker만 켜고 끈다. Y는 reward 패널만 켜고 끈다.
- `--no-rl-grasp-markers`로 표시를 끈다. `--rl-grasp-marker-radius 0.008`로 기준 구만 키울 수 있다.
- 실제 3D 도형이라 손가락/판 뒤에 가려질 수 있다. 각도를 바꿔 보거나 offset을 확인한다.
  목표 큐브는 판 두께를 정확히 반영하므로 서로 매우 가깝게 보일 수 있다.
- episode 종료 후 marker는 자동 reset된 현재 장면을 따라간다. 패널에 남은 종료 직전
  reward snapshot과 혼동하지 않는다.

구현은 `rl/debug/grasp_markers.py`다. 코드·CPU 기하 테스트만 수행하며 실제 VR 합성은 사용자가 확인한다.

정면에서 약간 아래, 눈앞 28cm의 머리 고정 텍스트 패널에 모든 활성 reward 항목을
약 10Hz로 표시한다. 카메라 패널은 기존처럼 위쪽에 둔다. 표시기는 reward 텍스트를
불투명 RGBA 이미지로 만들고, 기존 카메라 패널과 같은 XR 레이어·GPU 이미지 경로로
전송한다. 일반 `ui.Label`의 생성 여부만으로 VR 표시 성공을 판단하지 않는다.

- 각 항목: `raw × weight × control_dt`, 즉 **실제 해당 스텝의 보상 기여도**.
- `TOTAL`: 해당 스텝의 전체 보상. `RETURN`: 이번 시도의 누적 보상.
- 상승 높이(cm), 유지 시간(s), 좌/우 파지 판정, 좌/우 flap 목표 거리(cm).
- 정지 중에는 마지막 스텝 값이며 새 보상이 발생하지 않는다.

예를 들어 파지를 유지하면서 높이가 3→4cm로 증가하면 `lift: +0.83333`이다.
변화량 보상은 내부에서 dt로 나눈 후 RewardManager에서 곱하므로 스텝 주기에 이중으로 비례하지 않는다.
최초 유효 파지의 `flap_contact`는 +3이고, 유지/재파지 시에는 0이다.
성공 보너스 weight=150은 실제 성공 스텝에 `success: +150.00000`으로 표시한다.
TensorBoard `Episode_Reward/*`의 에피소드 정규화 값과 단위가 다르다.
0.1초보다 짧은 비종료 접촉은 화면 갱신 사이에 지나갈 수 있다. 이 모드는 전체
시계열 저장기가 아니며, 종료 보너스는 별도로 마지막 값을 유지한다.

접근 → 접촉 → 파지 → 들어 올리기 → 유지 순으로 `reaching`, `flap_contact`,
`lift`, `holding`이 변하는지 본다. `stable_grasp`는 기본 비활성화했다. 밀기·충돌 시에는
`prelift_disturbance`도 함께 확인한다. 현재 기본 preset의 `collision` 보상 항은 제거했고
장애물 힘은 진단용으로만 표시한다. 패널의 `Collision termination/penalty: OFF`로 확인한다.
보상은 양손에 각각 독립적으로
배분되는 것이 아니라 task 전체에 계산되며, 현재 기본 성공 조건은 **오른손 한 손 파지**다.

`orientation`은 오른손이 파지 전 면에서 10cm 이내일 때 정렬 점수의 개선/악화에 주는
signed 보상이다(weight 0.5). 같은 각도를 유지하거나 flap 후보가 전환된 순간에는 0이다.
`Alignment delta L/R`, `Lift delta (normalized)`, `Grasp credit used / award this step`으로
차분과 일회성 지급 여부를 확인한다. credit은 B/reset에서만 다시 사용할 수 있다.
각 flap 진단의 `axisErr`는 보정된 두 손가락 점 연결축과 flap 법선의
최소 각도(0~90°)다. 0°는 평행 정렬이며, 실제 손가락 패드 방향과의 일치까지
보장하는 값은 아니다. [닫힘 축 확인 절차](RL_RIGHT_HAND_PICK.md)를 함께 따른다.

### 들어 올렸는데 성공하지 않을 때

패널 **맨 위**의 `CHECK`는 아직 만족하지 않은 성공 전제조건이다. `FAIL`에는
실제 종료 원인(장애물 충돌·낙하·영역 이탈·안착 실패 등)을 별도로 표시한다.

| 표시 | 의미 |
|---|---|
| `grasp` | 지정 오른손 flap 파지가 인정되지 않음 |
| `height` | 안착 기준 상승 높이가 6cm를 넘지 못함 |
| `tilt` | 기울기가 40° 기준을 넘음 |
| `hold` | 파지·높이·기울기 조건의 연속 유지가 0.5초에 미달 |
| `initial_wait` | 초기 0.5초 대기가 끝나지 않음 |

pick 성공에서는 속도와 잔여 접촉력 제한을 사용하지 않는다. 기본
`collision_constraints_enabled=False`에서는 장애물 충돌 종료와 collision 비용도 없다.
True로 다시 켠 경우에만 20N 초과로 실패한다. 낙하, 영역 이탈 등 남아 있는 실패는
성공보다 우선한다. 기본 `reset_settle_timeout=0` 모드는
0.5초 고정 대기 후 기준 높이를 저장하며, 안착 타임아웃으로 실패하지 않는다.

패널은 기존 위치에서 머리 기준 왼쪽으로 5cm 이동했다. 상단에 별도 큰 글씨 영역을 두어
상세 reward/marker 텍스트가 길어져도 실패 이유와 성공 체크리스트 크기가 줄지 않는다.
`FAILED: OBSTACLE COLLISION` 등 실패 이유와 측정 force/기준을 표시하며,
`NEED:`(노랑)는 부족한 조건, `OK:`(초록)는 만족한 조건이다. 높이·기울기·유지 시간은
현재값/목표값을 함께 표시한다. 실패·성공·시간 초과 시 패널을 자동으로 다시 표시한다.
자동 reset 후 장면 대신 **종료 직전 샘플**을 계속 표시하고, B로 새 시도를 준비한다.

`R flap_right`와 `R flap_left` 각각의 `raw`는 현재 샘플의 최초 파지 조건, `held`는 제한된
접촉 유예까지 적용한 유지 판정이다. `region=11`은 두 손가락 접촉점이 모두 허용
영역 안에 있다는 뜻이고, `opp=1`은 양면 배치, `F=a/b N`은 해당 flap 접촉력이다.
`gap=...s`는 **접촉 누락 시간**이며 손가락 간격이 아니다. 유지 중 `raw=0, held=1`은
작은 미끄러짐·힘 변화·짧은 누락 때문에 발생할 수 있다. `slip`은 마지막 유효 접촉 대비
손가락 중점 이동, `open+`는 손가락 간격 증가량(mm)이다. 유효 접촉이 계속 확인되면
원점 이동만으로 해제하지 않으며 기준을 갱신한다. 이 제한은 접촉 누락 유예에만 적용한다.
최초 파지 전에는 둘 다 0이다.

물리가 진행 중이면 상승 여부와 무관하게 터미널에도 `[RL GRASP] blocked=...`
진단을 초당 한 번, 종료 스텝에는 즉시 출력한다. 문제가 계속되면 이 줄을 전달하면 된다.
disturbance는 초기 선반 위치·자세 비용과 운동 비용을 분리했다. 1cm 이상 상승하면
전자만 해제되며, 파지 없이 움직이는 박스에는 운동 비용이 남는다. 상세 식과 유지
판정 허용치는 [flap-pick 정의](RL_FLAP_PICK.md)에 있다.

파지 유지 수정 후에는 검사 프로세스를 재시작한다. 이 수정은 학습과 검사 모드가
공유하는 RL 판정이다. 후보 ID observation과 기본 8-action 계약도 변경되어, 수정 전 checkpoint를
직접 재개하는 대신 새 학습을 시작해야 한다.

## 기존 수집기와 동일한 부분 / RL에 맞추는 부분

- 동일: OpenXR 장치·연결, scaled controller 위치/회전 매핑, position gain,
  URDF IK·arm response·orientation weight, head 위치 기준 시점, render quality,
  XR resolution scale, desktop render 옵션.
- RL 유지: `configs/rl_pick_arms_only.py`, named initial pose, 선택된 action 순서(`0`은 오른팔 7+1,
  `1`은 all-joints)와
  증분 크기, actuator·마찰, 관측·reward·충돌·성공/실패 판정과 제어 주기.
  입력만 정책 대신 Quest/IK에서 만든다. 수집기의 별도 관절 구동값·중력 보상·
  self-collision guard를 추가하지 않는다. 따라서 손의 응답은 일반 수집기와 다를 수 있다.
- `0`에서는 base·waist·head를 고정하고 stick 이동을 사용하지 않는다. `1`에서는
  왼쪽 stick으로 base 전후·좌우, 오른쪽 stick 좌우로 base 회전, 상하로 torso를
  움직인다. 로봇 head action도 해제하지만 현재 HMD 방향을 head joint 명령으로 쓰지는 않는다.
- 기본 `active_arm="right"`에서는 왼팔·왼손도 고정된다. 왼쪽 컨트롤러의 X/Y 버튼은
  시점/패널 제어에 계속 쓰지만 왼팔 IK·왼쪽 trigger action은 생성하지 않는다.
  동작 중 tracking 검사는 HMD와 활성 오른쪽 컨트롤러만 필요하다.
- 기본 Quest overlay는 left/right wrist RGB 패널만 사용한다. head sensor는 기본으로
  생성하지 않으며 `--head-camera`를 넣었을 때만 PC camera preview에 추가된다.
  이 영상은 표시 전용이고 RL observation에는 들어가지 않는다. Depth·데이터 기록·
  공장 배경은 생성하지 않는다. `--control-hz`, `--episode-seconds`, `--arm-stiffness`,
  `--arm-damping`, recording 관련 옵션은 이 모드에서 적용하지 않는다.
- 지원 조합은 **controllers + scaled, absolute 또는 relative**다. `--controller-mapping absolute`를
  쓰면 `--absolute-orientation`(downward/pointing)도 그대로 적용되고, `AbsoluteControllerMapper`가
  aim pose로 방향을 계산한다. `--hand-switch`, hands, `--scene-config`,
  `--domain-randomization`은 거부한다.
  수집기의 `--auto-start`와 무관하게 A로 명시적으로 시작한다.

실제 로봇 안전 제어기가 아니다. USD 물체를 순간 이동시키지 않고 기존 RL action을
통해 물리적으로 움직여 접촉 보상을 검사한다.

## 설정 수정

- `configs/rl_pick_arms_only.py`: 초기 프리셋 이름, 손 선택, 성공 조건과 실험별 설정.
- `src/kuavo_isaaclab_scene/rl/managers/rewards.py`: reward 가중치.
- `src/kuavo_isaaclab_scene/rl/mdp/rewards.py`: 항목별 계산식.

실험별 가중치를 바꾸려면 config의 기존 `configure(env_cfg, agent_cfg)` 안에
`env_cfg.rewards.reaching.weight = 6.0`처럼 추가한다. 학습 중인 config를 바꾸기보다
복사본을 만들어 다음과 같이 선택하는 것을 권장한다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-config configs/my_reward_probe.py
```

설정 수정은 **검사 프로세스를 재시작**해야 반영된다. 자동 파일 reload는 없다.
로컬 초기 자세·gripper config를 읽으므로 기존 학습의 `manifest.json`/`env.yaml`과
다르면 그 checkpoint가 사용한 환경의 완전한 재현은 아니다.

구현은 `rl/debug/`에 격리되어 있고 학습 runner에서는 import하지 않는다.
RewardProbe는 reward 계산 후, 자동 reset 전에 이미 계산된 값을 읽을 뿐
RewardManager.compute를 추가 호출하지 않는다. 변경 후 시뮬레이터 실행/VR 시각 검증은
수행하지 않았으며 실제 패널 위치·크기와 조작성은 사용자가 확인한다.

## 디버깅 패널 표시값 수정

표시만 바꾸는 것과 학습에서 쓰는 reward를 바꾸는 것은 별개다. 아래 경로는
저장소 루트 기준이며, 설정·코드 수정 후 검사 프로세스를 종료하고 재시작한다.

| 수정 목적 | 파일 / 위치 |
|---|---|
| 본문 항목, 순서, 이름, 소수점 | `src/kuavo_isaaclab_scene/rl/debug/reward_report.py` → `format_report()` |
| 큰 성공/실패 제목, 부족한 조건 | 같은 파일 → `reward_summary()`, `FAILURE_LABELS` |
| 새로운 물리 상태값 가져오기 | `src/kuavo_isaaclab_scene/rl/debug/reward_recorder.py` → `RewardProbe.record_post_step()`의 sample dictionary |
| 최근 reaching 합계 시간창 | `reward_report.py` → `ReachRewardSummary(window_s=0.5)` |
| marker/collider 설명줄, HUD 갱신 간격 | `src/kuavo_isaaclab_scene/rl/debug/quest_reward.py` → `report` 조립 부분, `last_hud`의 `.1`초 조건 |
| 글꼴·색상·텍스처 크기·패널 위치 | `src/kuavo_isaaclab_scene/display/xr_reward_panel.py` → `reward_image()`, `QuestRewardPanel.__init__()` |

### 항목을 줄이거나 순서 바꾸기

기본 `format_report()`는 `sample["terms"]`의 모든 활성 reward 항목을 표시한다.
예를 들어 이 함수의 아래 줄을:

```python
lines += [f"{name}: {value:+.5f}" for name, value in sample["terms"].items()]
```

다음처럼 바꾸면 선택한 항목만 원하는 순서로 표시한다. reward 계산·학습 가중치는
바뀌지 않으며, `TOTAL`과 `RETURN`에는 숨긴 항목도 계속 포함된다.

```python
display_terms = ("reaching", "orientation", "lifted", "disturbance")
for name in display_terms:
    if name in sample["terms"]:
        lines.append(f"{name}: {sample['terms'][name]:+.4f}")
```

항목 이름은 실제 활성 manager key와 같아야 한다. 현재 출력 또는
`rl/managers/rewards.py`에서 확인한다. 파지 상세 설명이 너무 길면
`lines.extend(sample.get("grasp_debug", []))`를 제거하거나 일부만 표시한다.
큰 성공 조건 헤더는 별도로 `reward_summary()`에서 만들어지므로 본문을 줄여도 유지된다.

### 새로운 상태값 추가하기

예를 들어 손가락 벌어짐 변화도 보고 싶다면 아래처럼 추가한다.
`grasp_candidate_opening_m`은 **현재 절대 손가락 간격이 아니라 파지 기준 대비
opening 증가량(m)** 이므로 mm 표시를 위해 1000을 곱한다:

```python
# RewardProbe.record_post_step(): sample dictionary 안
"opening_delta_mm": (1000 * t.grasp_candidate_opening_m[0]).detach().cpu().tolist(),

# format_report(): sample이 None이 아닌 else 블록 안
lines.append(f"Opening delta L/R x flap (mm): {sample.get('opening_delta_mm', [])}")
```

기존 command/sensor의 캐시를 읽어야 한다. 표시를 위해 reward 함수나
`RewardManager.compute()`를 다시 호출하면 안 된다. progress/latch 같은 상태가
추가로 갱신될 수 있기 때문이다. recorder는 **자동 reset 이전** 값으로 snapshot을
만들며, 패널은 이 snapshot을 읽는다. 종료 후 장면이 초기화되어도 마지막 패널 값은
종료 직전 상태를 뜻한다.

### 위치·글자 크기와 실제 보상 변경

패널 위치는 `QuestRewardPanel.__init__()`의
`position=(-5., -4., -28. ...)`이며 단위는 cm, head-relative 좌표다.
X를 더 음수로 바꾸면 더 왼쪽으로 간다. `reward_image()`의 `_font(36)`은 제목,
`_font(24)`는 조건, `_font()`는 본문이다. 폰트만 키우면 줄 높이도 함께 조절한다.
행이 많으면 본문이 전체 높이에 맞게 축소되므로, 우선 표시 항목 수를 줄이는 것이 좋다.
텍스처 크기를 바꾸려면 `reward_image()`의 기본 width/height와 panel 생성자의
width/height를 일치시킨다. `pixels_per_cm`은 패널의 물리적 표시 크기에 영향을 준다.

실제 점수는 `configs/rl_pick_arms_only.py`의 `configure()` 또는
`rl/managers/rewards.py`에서 weight를, `rl/mdp/rewards.py`에서 계산식을 수정한다.
패널의 `terms`는 이미 **raw × weight × control dt**가 적용된 step 기여도다.
표시 코드에서 weight/dt를 다시 곱하지 않는다. Reach/Alignment state score는
진단용 상태 점수이며 실제 step reward가 아니다.

Mode 0/1 패널은 단일 박스 flap-pick 및 pick_place reward를 표시한다. Mode 2는 같은
XR panel에 multi-box 배치와 선택한 box의 read-only pose shadow reward 및 raw contact를
표시한다. 아직 승인되지 않은 접촉 임계값·성공/충돌 event·phase transition은 계산하거나
제어에 적용하지 않는다.

## 패널이 안 보이거나 오류 로그가 나오는 경우

- Y/H를 누르면 `[RL REWARD] Reward panel ON/OFF`와 `[XR REWARD TEXTURE]`가 출력된다.
  `ON`, `layer=True`, `panel=True`, `widget_ready=True`인데도 안 보이면 시점 방향이나
  가림 문제일 수 있다. 먼저 카메라 패널을 끄고 확인하고, 필요하면 기존
  `--xr-overlay-forward-axis +z`로 방향을 비교한다. 이는 원인이 확정된다는 뜻은 아니다.
- `widget_ready=False`가 계속 나오면 XR UI 생성 문제다. 이 상태 로그와 먼저 발생한
  오류를 함께 전달한다. 버튼 수신 로그만으로 화면 표시 성공을 판단하지 않는다.
- `[XR REWARD TEXTURE] uploads=...`는 GPU 이미지 업로드 횟수이며, 예전 `[XR PANEL]`
  로그만 나온다면 변경 전 프로세스다. 수집기를 재시작해야 한다. 업로드 성공 역시
  실제 헤드셋 합성까지 검증한 값은 아니다. 투명하지 않은 테두리/배경까지 전혀
  보이지 않으면 이 로그와 카메라 패널의 표시 여부를 함께 전달한다.
- RGBA 텍스트 생성에는 Pillow 10.1 이상을 사용한다(현재 로컬 Isaac 환경에 설치됨).
  다른 PC에서는 프로젝트의 `rl` extra에 포함된다. RGB 카메라 센서를 끈 상태에서도
  작은 desktop viewport 업데이트를 유지해 UI 텍스처가 갱신되도록 한다.
- `bind_event_generator ... token: press`가 `[Error]`로 출력되어도 이후
  `[BUTTON] ... pressed`가 나오면 해당 버튼 콜백은 동작한 것이다. 이 메시지만으로
  연결 실패나 프로세스 중단으로 판단하지 않으며, SDK 로그를 전역으로 숨기지 않는다.
- `Body must be non-kinematic`는 무시할 로그가 아니다. 검사 모드의 고정 conveyor
  표면에 대한 불필요한 zero-velocity reset을 차단했다. 박스의 동역학이나 RL 학습
  runner는 변경하지 않는다. 같은 오류가 계속되면 다른 호출/물체도 확인해야 한다.
