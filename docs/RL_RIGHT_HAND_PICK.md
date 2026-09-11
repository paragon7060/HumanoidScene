# 오른손 전용 flap 파지: 구현과 직접 튜닝

현재 실험 설정은 `configs/rl_pick_arms_only.py`, contract는 `flap_grasp_revision=3`이다.
이번 목표는 오른손으로 박스를 집어 안착 높이보다 6cm 넘게 들고 0.5초 유지하는 것이다.
랙 밖 인출·컨베이어 이동은 아직 포함하지 않는다. 베이스·허리·머리와 왼팔·왼손은
`quest_ready_02`의 초기 관절값에 고정한다. 일반 Quest 수집기는 변경하지 않으며,
`--rl-reward-debug`만 이 RL 설정을 공유한다.

## 1. Action과 고정 관절

```python
active_arm="right"
grasp_hand="right"
required_grasp_hands=1
```

Action은 `zarm_r1_joint` ~ `zarm_r7_joint` 7개와 오른쪽 그리퍼 1개, 총 8개다.
팔 목표는 현재 누적 목표에 `clamp(action,-1,1) * 0.02rad`를 더한다. 그리퍼는
signed target에 `action * 0.08`을 누적하고 [-1,1]로 제한한다. 0은 이전 목표 유지다.
왼쪽에 dummy action을 남기지 않는다.

`FixedBody`는 기존 몸통/머리/다리/바퀴 외에 `zarm_l[1-7]_joint`, `l_.*_joint`를
선택해 왼팔과 구동/수동 손가락 linkage를 함께 제한한다. 매 reset 후 캡처된 초기값
주변 ±0.0001rad로 물리 joint limit을 좁히고 각 물리 substep에 같은 PD 목표를 유지한다.
매 스텝 joint pose를 순간이동시키는 방식이 아니다. 힘에 의한 수치 오차까지 0은 아니다.
현재 이 단일 팔 설정은 기존 RL이 지원하는 로봇 일체형 two-finger용이다.

양팔을 다시 열려면 `active_arm="both"`로 바꾼다. 오른손 보상만 유지할 수 있으며,
양손 파지가 반드시 필요하면 `required_grasp_hands=2`도 설정한다.

## 2. 오른손의 접근 거리

```python
grasp_flaps=("flap_right", "flap_left")  # 손별 배정이 아니라 공유 후보 목록
```

매 제어 스텝, 오른손 tool reference를 두 flap 각각의 실제 rigid link 좌표계로
변환한다. composed USD의 중심·half size를 사용해 **두 넓은 면 전체**에서 최근접점을
계산하고 월드 좌표계로 돌려놓는다. 회전·spawn scale·병렬 env offset을 반영한다.
면의 테두리도 포함하지만 판 내부를 거리 0인 부피로 취급하지는 않는다.

```text
d_right = min(distance(tool_right, surface(flap_right)),
              distance(tool_right, surface(flap_left)))
score = exp(-12 * d_right)
progress = score - 직전 스텝 score  # 파지 전, 연속 유효 스텝만
reaching raw = progress / control_dt
실제 스텝 기여도 = 4 * progress
```

최초 기준이 10cm인 예: 5cm까지 접근하면 누적 약 +0.99047, 5cm에서 정지하면 0,
10cm로 후퇴하면 -0.99047, 다시 5cm로 접근하면 +0.99047이다. 4cm까지 더 접근하면
추가 약 +0.27989를 받는다. 같은 활성 구간 안에서 여러 스텝으로 이동해도 같은 끝 거리면
할인 전 누적 보상은 같다. 이는 단순 signed 차분이며 PPO gamma=0.99에 맞춘 엄밀한
potential-based shaping은 아니다. 할인된 왕복 수익까지 완전히 상쇄한다고 보장하지 않는다.

왼손 거리는 측정/표시하지만 기본 오른손 reward에는 넣지 않는다. 정렬도는 별도
orientation 항으로 유지한다. 이 변경은 현재 flap-pick reaching에만 적용하며,
approach_rack/navigation·orientation·flap_contact·stable_grasp·lift·시간 비용은 변경하지 않는다.
정지 상태에서 **전체** reward가 반드시 음수가 되는 것은 아니다.

직전 점수는 환경별·손별로 관리하고, 초기 대기 종료/리셋/대상 박스 변경 때는 보상 없이
현재 거리로 기준을 잡는다. 같은 박스의 가까운 flap 후보가 바뀌어도 기록을 초기화하지 않는다.
기존 contact latch의 `hand_grasp_flags`를 사용해 **파지를 획득한 스텝부터 유지 중에는
해당 손의 reaching을 0**으로 한다. 파지를 잃은 첫 스텝도 보상 없이 기준만 잡고,
그다음 스텝부터 signed 보상을 재개한다. lifting 중 정상적인 상대 위치 변화는 감점하지 않는다.
대상 박스의 움직임으로 거리가 줄어든 경우도 기하학적 진전으로 측정된다.

`rl/mdp/reach_progress.py`가 직전 점수와 변화량을 관리한다. `reaching_history` observation은
양손의 최신 활성 점수 2개(다음 스텝의 기준)이며, 파지 중/비활성일 때는 0이다.
차원은 같지만 의미가 달라 **이전 checkpoint는 그대로 재개하지 않는다.**
reward contract에는 `flap_signed_pregrasp_progress_v2`를 기록한다.
VR 패널은 `Reach signed progress L/R`, `Reach baseline score L/R`, `Reach tracking L/R`를
표시한다. 왼손 tracking이 켜져 있어도 기본 보상에는 오른손만 들어간다.
`Reaching last 0.5s | episode`는 모든 제어 스텝의 **실제 가중 보상**을 합산한 값이다.
짧은 양/음 보상이 10Hz 표시 사이에 지나가도 최근 합계·누적에 포함된다. 서로 상쇄되면
합계는 0일 수 있다. pause 시 시간이 멈추고 마지막 값을 유지하며, B/reset 시 합계를 비운다.
S200062의 거리 기준은 [보정된 closed endeffector_center](ENDEFFECTOR_CENTER.md)다.
기존 `tool_offset=(0,0,-0.12)`는 이 모델에서 사용하지 않는다. 두 점은 보정 파일에서 수정하며,
변경 후 재시작해야 한다. 접촉 판정의 힘·접촉점은 계속 실제 filtered contact를 쓴다.

### 약한 orientation 보상과 닫힘 축 확인

거리 보상에 곱하지 않고 `orientation` 항을 별도로 더한다. 기본은 오른손만,
안착 완료 후 pick 단계·파지 전·면 거리 10cm 이내에서 활성화된다.

```text
a = abs(dot(두 손가락 link 원점을 잇는 단위축, 최근접 flap의 단위법선))
orientation = 0.5 * clamp(1-d/0.10,0,1) * a² * 비파지 * dt
```

30Hz 최대 기여도는 +0.01667로 reaching 최대 +0.13333의 1/8이다.
법선 부호를 구분하지 않고, 법선 주위 회전은 제한하지 않는다. 파지 유지 조건이나
성공 조건에 새로운 orientation 임계값을 추가하지 않는다. observation/action 차원은 그대로다.
`configure()`의 `env_cfg.rewards.orientation.weight`와
`env_cfg.rewards.orientation.params["distance_threshold"]`를 수정할 수 있다.
축 검증 전 이 항을 끄려면 weight를 0으로 설정한다.

현재 로컬 코드의 표시 기능을 이용해 다음과 같이 확인한다. Runtime/web 연결 후 검사
프로세스만 실행하며, 다른 수집/검사 프로세스를 중복 실행하지 않는다.

```bash
./quest_collector.sh collect --rl-reward-debug --desktop-render \
  --rl-grasp-markers --no-rl-grasp-targets --rl-collision-view \
  --rl-grasp-finger-offsets 0 0 0 0 0 0
```

1. 청록/파랑 구가 각각 `r_f_finger`, `r_b_finger`의 **link 원점**이다.
   offset이 0이므로 이 두 구를 잇는 방향이 현재 reward의 닫힘 축이다.
2. collider 윤곽선과 두 안쪽 패드 면을 옆/위 등 서로 다른 시점에서 확인한다.
   두 구를 잇는 방향이 양쪽 패드 면에 거의 수직이어야 한다.
   초록 면은 자동 선택된 후보일 뿐이므로 실제 패드인지 직접 확인한다.
3. 손목 자세를 유지하고 그리퍼를 열고 닫아 본다. 두 원점 사이 간격이 변하는 동안
   그 방향이 실제 패드가 마주 보는 방향과 계속 일치하는지 본다. linkage가 회전하면
   원점 연결 방향이 실제 접촉면 법선과 어긋날 수 있다.
4. flap을 정상적으로 집을 수 있는 자세에서 해당 후보의 `axisErr`가 0°에 가까운지
   확인한다. 90°이면 코드상 닫힘 축이 flap 면에 평행한 것이다.

`axisErr = acos(abs(dot(axis,normal)))`이며 0~90°다. 물리 스텝 이후 패널/터미널에
두 flap 각각 표시한다. 일시정지 중에는 마지막 측정값이다. `axisErr`는 원점 연결축과
flap 법선의 관계만 나타내므로 **패드 면과 축이 맞는지 자체를 입증하지 않는다.**
맞지 않으면 임계값을 완화하기보다 검증된 패드 법선 또는 보정된 두 기준점으로
닫힘 축 정의를 변경해야 한다. 표시용 offset/보정 JSON 변경은 reward에 자동 적용되지 않는다.

## 3. 최초 파지는 어떻게 판정하는가

각 손가락 센서는 두 flap을 따로 읽는다. 필터 순서는 `box0/flap_right`,
`box0/flap_left`, `box1/flap_right`, `box1/flap_left`다. 각 후보 flap에 대해:

1. 두 손가락의 평균 접촉점이 모두 finite여야 한다. 접촉 없음/NaN은 false다.
2. 두 점이 해당 flap의 실제 로컬 bounds + `flap_contact_margin=0.004m` 안에 있어야 한다.
3. 같은 flap에 대한 각 손가락 접촉력 norm이 `grasp_force=0.20N`을 **초과**해야 한다.
4. 두 접촉점이 판 중심 기준 반대쪽에 있거나, 두 손가락 link 원점이 반대쪽에 있어야 한다.

현재 `flap_contact_region="surface"`라 전체 판이 유효하다. `"top_band"`로 바꾸면
상단 `flap_top_band=0.030m`만 새 파지를 인정한다. 거리 reward는 이 옵션과 무관하게 전체 면이다.
`flap_grasp_depth`는 이전 상단 점 목표의 호환 필드로 남아 있으며 현재 면 거리를 바꾸지 않는다.

위 네 조건을 모두 만족하면 그 후보의 `raw=1`이다. 오른손이 어느 후보 하나를 잡으면
파지로 인정하지만, 손가락 A가 flap_right, 손가락 B가 flap_left에 닿은 힘을 합쳐
하나의 파지로 인정하지 않는다. 박스 본체 접촉만으로도 인정하지 않는다.
평균 접촉점과 link 원점은 접촉 패치 전체나 force-closure를 완전히 설명하지 못하는
근사 판정이다. 시각적 파지와 다른 경우 아래 로그를 보고 조건을 조정해야 한다.

## 4. 파지 유지와 해제

손×후보별 latch가 있다. 최초 파지 시 해당 flap 좌표계에서 손가락 두 link 원점의
중점과 간격을 저장한다. 유지 중에는:

- 같은 후보의 두 접촉점이 허용 영역에 있고, 각 힘 > 0.10N, 양면 배치가 유지되면 정상 접촉.
- 정상 접촉이 있으면 link 원점의 중점 이동·간격 증가량만으로 해제하지 않는다.
- 정상 접촉이 확인될 때마다 해당 중점·간격을 최신 기준으로 갱신한다.
- 정상 접촉이 빠진 경우에만 마지막 기준 대비 `slip <= 0.020m`, `open+ <= 0.008m`일 때 0.10초 미만까지 유예.
- 접촉 누락 중 위 범위를 벗어나면 즉시 해제. 긴 누락 뒤에는 다시 최초 파지 조건이 필요.

이 규칙은 `flap_hold_revision=2`로 checkpoint 계약에 기록한다. observation/action 차원은
변경하지 않았지만 기존 유지 판정과 의미가 달라 이전 계약의 자동 재개는 거부한다.
`slip`/`open+` 로그는 이제 최초 파지가 아닌 **마지막 유효 접촉 대비** 값이다.
정상 접촉이 있는 스텝에서는 기준을 갱신하므로 보통 0으로 표시된다.

`top_band`만 유지 접촉 영역을 아래로 2cm 확장하며, 전체 면 모드에서는 판 바깥으로
확장하지 않는다. 유지 중 근접 목표가 다른 flap으로 바뀌어도 잡고 있는 flap의 이력을
다른 후보로 넘기지 않는다. 박스를 attach하거나 높이만 보고 파지를 인정하지 않는다.

오른손 `held=1`, 상승 >6cm, 기울기 <40°가 초기 대기 완료 후 0.5초 이어져야 성공이다.
pick 성공의 선속도·각속도·잔여 손가락 접촉력 제한은 제거했다. 실제 파지 판정용 flap
접촉력과 그 밖의 보상/관측은 유지한다. 이후 carry/place 단계의 안정성 조건은 별도다.
잔여 힘은 `norm(net_force - sum(두 허용 flap의 force vector))`라는 근사값이다.
현재 기본 설정은 `collision_constraints_enabled=False`로 장애물 충돌 실패와 `collision`
보상 항 전체(장애물 힘 + 잔여 손가락 힘)를 끈다. 접촉 물리·센서·파지 판정은 유지한다.
낙하·영역 이탈 등 나머지 failure는 여전히 성공보다 우선한다. 기본 설정은 0.5초 고정 대기 후
시작하며 `settle_timeout` 실패는 없다.

충돌 제한을 다시 켜려면 `configs/rl_pick_arms_only.py`의 `configure_task()`에서
`collision_constraints_enabled=True`로 변경한다. 그러면 >20N 충돌 실패와 기존 -2 가중치
collision 비용이 복구된다. 일반 whole-body task 기본값은 바꾸지 않았다.
학습과 VR reward 검사는 같은 설정을 사용한다. 실행 중인 프로세스에는 반영되지 않으므로
재시작해야 하며, checkpoint 계약도 변경되어 새 실험을 시작해야 한다.
무충돌 정책을 보장하지 않는 시뮬레이션 학습 전용 설정이다.

## 5. disturbance 완화

가중치 -4 → -0.25, 전체 raw 상한 1을 둔다. 이 항만으로는 30Hz에서
스텝당 -0.00833보다 더 큰 음수가 발생하지 않는다. 다른 충돌/실패 벌점까지 제한하는 것은 아니다.

| 물리량 | 비용 0인 deadband | 초과량 정규화 scale |
|---|---:|---:|
| 초기 대비 수평 변위 | 5mm | 5cm |
| 수평 속도·비파지 수직 속도 | 0.02m/s | 0.20m/s |
| 각속도 | 0.10rad/s | 1.0rad/s |
| 초기 대비 회전 | 5° | 30° |

각 비용은 `C(x)=max(x-deadband,0)^2 / scale^2`다. 다음은 코드와 같은 순서다.

```text
P = min(C(수평변위),4) + min(0.5*C(회전),4)
M = min(0.25*C(수평속도),4) + min(0.25*C(각속도),4)
    + min(0.25*C(수직속도)*비파지,4)
w = clamp(1 - max(상승높이,0)/0.01, 0, 1)
D = P*w + M*max(w, 비파지)
유효 파지면 D *= 0.25
raw = min(D,1)
reward = -0.25 * raw * dt
```

안착 높이에서 1cm 이상 올라가면 초기 위치/자세 벌점은 없다. 비파지 상태로 던진 박스에는
운동 비용이 남지만, 진짜 파지 중 수직 상승에는 수직속도 벌점을 주지 않는다.
`stable_grasp`는 같은 deadband/scale을 사용한 `grasped * exp(-수평속도비용-각속도비용-수평변위비용*w)`다.
작은 떨림으로 이 보상도 급격히 감소하지 않게 한다. 초기 안착 완료 전에는 둘 다 비활성화한다.
진단을 위해 disturbance 자체를 잠시 끄려면 config의 `configure()`에서
`env_cfg.rewards.prelift_disturbance.weight = 0.0`으로 바꿀 수 있다.

## 6. VR에서 직접 확인

실제 collider·안쪽 면 후보·접촉점·힘 화살표는
[collision 표시 안내](RL_QUEST_REWARD_DEBUG.md#실제-collider안쪽-면-후보접촉점힘-표시)를 참고한다.
이는 기존 수동 offset marker와 별개이며 시뮬레이션 정보를 직접 사용한다.

두 기준점과 반대 면 목표점의 3D 표시 및 offset 보정 방법은
[Quest 검사 표시 안내](RL_QUEST_REWARD_DEBUG.md#오른손-기준점과-목표점의-3d-표시)를 참고한다.
이 marker는 새 reach 설계 미리보기이며 현재 reward와 파지 판정을 변경하지 않는다.

이미 Runtime/web이 연결되어 있다면 검사 프로세스만 재시작한다.

```bash
./quest_collector.sh collect --rl-reward-debug
```

시작 로그에서 `active_arm=right`, `actions=8`, `contact_region=surface`를 확인한다.
왼쪽 컨트롤러의 X/Y는 시점/패널 기능으로 계속 사용하지만 왼팔과 왼쪽 그리퍼는 움직이지 않는다.
패널 맨 위에 `CHECK`, `FAIL`과 두 후보 각각의 다음 정보가 나온다.

```text
R flap_right: d=2.8cm raw=1 held=1
region=11 opp=1 F=0.80/0.76N gap=0.000s
slip=2.0mm open+=0.2mm
```

물리 실행 중 터미널 `[RL GRASP]`에도 초당 한 번, 종료 시 즉시 같은 내용을 출력한다.
실제 VR 합성은 사용자가 확인한다. 표시가 안 되어도 터미널에서 판정 원인을 읽을 수 있다.

| 값 | 먼저 볼 것 / 조정할 곳 |
|---|---|
| F 중 하나가 0에 가까움 | 손가락 collider가 실제 그 flap에 닿는지. 시각 mesh 접촉만으로는 부족 |
| F는 있는데 region이 01/00 | 평균 접촉점, USD bounds/scale, `flap_contact_region`, `flap_contact_margin` |
| region=11, F>0.2인데 opp=0 | 두 손가락/접촉점의 판 법선 부호. `opposed_jaws()` 근사가 모델과 맞는지 |
| raw=1인데 held=0 | 이전 후보가 아직 유지 중인지. 유효한 같은 후보의 접촉은 원점 이동만으로 해제하지 않음 |
| held=1인데 NEED: Hold | 높이/기울기를 포함한 조건을 연속 0.5초 유지해야 함 |
| FAIL=obstacle_collision | 현재 OFF. collision_constraints_enabled=True일 때만 >20N 실패 |

디버거에서는 `FlapGrasp.measure()`의 `t.grasp_candidate_contact_local`을 확인할 수 있다.
shape는 `[env, hand, candidate, jaw, xyz]`이고 값은 flap 로컬 m다.
`self.centers[t.active_box]`, `self.halves[t.active_box]`와 비교한다.
`t.grasp_candidate_force/region/raw/held`, `t.grasp_candidate_slip_m/opening_m`도 같은
hand/candidate 순서다. 후보 0/1은 config tuple 순서, hand 0/1은 로봇 왼손/오른손이다.

## 7. 수정 파일과 checkpoint

| 수정 내용 | 파일 |
|---|---|
| 초기 프리셋·활성 팔·후보·접촉 영역·유예·disturbance 수치 | `configs/rl_pick_arms_only.py` |
| 필드 정의·validation | `src/kuavo_isaaclab_scene/rl/tasks/specs.py` |
| 센서별 두 flap 필터 | `src/kuavo_isaaclab_scene/rl/scenes/sensors.py` |
| 면 거리·후보별 접촉 판정 | `src/kuavo_isaaclab_scene/rl/mdp/flap_grasp.py` |
| 파지 유지 이력 | `src/kuavo_isaaclab_scene/rl/mdp/grasp_contact_latch.py` |
| disturbance 계산 | `src/kuavo_isaaclab_scene/rl/mdp/grasp_stability.py` |
| reward 함수 / 가중치 기본값 | `src/kuavo_isaaclab_scene/rl/mdp/rewards.py`, `rl/managers/rewards.py` |
| 고정 관절 / action | `src/kuavo_isaaclab_scene/rl/mdp/body_lock.py`, `rl/mdp/actions.py` |
| 성공·실패 조건 | `src/kuavo_isaaclab_scene/rl/mdp/commands.py` |
| VR 정보 수집·텍스트 구성 | `src/kuavo_isaaclab_scene/rl/debug/reward_recorder.py`, `reward_report.py` |

관측에 양손 접촉 대표 후보 one-hot 4개를 추가했다. 최근접 면 상대 위치와 파지 접촉은
서로 다른 후보일 수 있어, 접촉력/유지 상태가 어느 후보의 값인지 구별하도록 한다.
기존 순간 파지 flag 2개·누락 시간 2개도 유지한다. 반면 왼팔 action과 actuator 목표가
빠져 전체 observation/action 차원이 바뀐다. 모델별 수치를 임의로 고정하지 않고 runner가
실제 차원을 사용한다. **revision 2 및 이전 checkpoint는 직접 재개하지 않는다.**

VR 확인 후 새 학습은 `./train_flap_pick.sh`를 `--checkpoint` 없이 실행한다.
이번 변경은 코드·CPU tensor 테스트이며, 시뮬레이터/VR 실행이나 성공률 검증은 아니다.
고정된 왼손의 수동 지지까지 금지하는 별도 판정은 추가하지 않았다. 오른손 파지는 필수지만
왼손이 초기 배치상 우연히 박스를 받치는 상황은 배치와 로그로 확인해야 한다.
