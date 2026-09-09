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
reaching raw = exp(-12 * d_right)
실제 스텝 기여도 = 4 * raw * (1/30)
```

| 면까지 거리 | raw | 30Hz 스텝 보상 |
|---|---:|---:|
| 20cm | 0.0907 | +0.01210 |
| 10cm | 0.3012 | +0.04016 |
| 5cm | 0.5488 | +0.07317 |
| 0cm | 1 | +0.13333 |

왼손 거리는 측정/표시하지만 reward에 넣지 않는다. 기존 정렬도 배율을 제거해,
다른 상태가 같을 때 가까울수록 반드시 더 큰 접근 보상을 받는다. 정렬도는 observation에 남는다.
거리 기준은 손가락 mesh 전체가 아니라 end-effector의 `tool_offset=(0,0,-0.12)` 지점이다.
화면의 손과 거리값이 어긋나면 `tool_offset`을 조정해야 한다. 접촉 판정은 이 proxy가 아니라
실제 filtered contact를 쓴다.

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

- 중점 이동 `slip <= 0.020m`.
- 간격 증가 `open+ <= 0.008m`.
- 같은 후보의 두 접촉점이 허용 영역에 있고, 각 힘 > 0.10N, 양면 배치가 유지되면 정상 접촉.
- 정상 접촉이 빠져도 상대 배치가 위 범위 안이면 0.10초 미만까지 유예.
- 너무 벌어지거나 미끄러지면 즉시 해제. 누락이 길어지면 다시 최초 파지 조건이 필요.

`top_band`만 유지 접촉 영역을 아래로 2cm 확장하며, 전체 면 모드에서는 판 바깥으로
확장하지 않는다. 유지 중 근접 목표가 다른 flap으로 바뀌어도 잡고 있는 flap의 이력을
다른 후보로 넘기지 않는다. 박스를 attach하거나 높이만 보고 파지를 인정하지 않는다.

오른손 `held=1`, 상승 >6cm, 기울기 <40°, 선속도 <0.08m/s, 각속도 <0.35rad/s,
두 허용 flap 외 손가락 잔여 접촉력 <10N이 준비 완료 후 0.5초 이어져야 성공이다.
잔여 힘은 `norm(net_force - sum(두 허용 flap의 force vector))`라는 근사값이다.
장애물 접촉 >0.1N 등 failure는 여전히 성공보다 우선한다.

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
| raw=1인데 held=0 | 초기 대비 slip/open 제한 초과 또는 이전 후보가 아직 유지 중인지 |
| held=1인데 CHECK에 speed | 0.08m/s, 0.35rad/s 기준. 잠시 멈춰 유지 |
| FAIL=obstacle_collision | 모든 robot link의 주변 장애물 센서 >0.1N. 이번 완화 대상이 아님 |

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
