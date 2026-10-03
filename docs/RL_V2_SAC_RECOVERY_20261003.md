# V2 grasp SAC: 장기 학습 붕괴와 복구 실험 (2026-10-03)

## 종료된 실험 결과

10/02 초반의 연속 성공은 장기 SAC 학습에서 유지되지 않았다. 예정된 횟수를
모두 수행한 뒤 종료했으며, 런타임 오류나 OOM으로 중단된 실행은 아니다.
아래 결과는 종료된 `results.json`과 각 실제 물리 실행의 `metrics.json`에서 집계했다.

| 실험 | 실제 훈련 | 마지막 모델 고정 평가 | 안전 위반 / 시간 초과 |
|---|---|---|---|
| GPU3 목표 SAC, gamma0.99 | 9/48 성공 | 0/12 성공 | 훈련28/11, 평가11/1 |
| GPU0 목표 SAC, gamma0.999 | 6/48 성공 | 0/12 성공 | 훈련16/26, 평가0/12 |
| GPU0 BC, optimizer 업데이트0 | 훈련 없음 | 10/12 성공 | 평가0/2 |

동일한12개 랜덤 box·초기 base XY/yaw 평가에서 **현재 BC가 마지막 SAC보다 좋다.**
SAC가 BC보다 우월하거나 다양한 크기·위 선반까지 성공했다고 주장하지 않는다.
종료된 세 suite의 최종 Drive 업로드 크기·MD5 검증은 완료됐다.

![실제 학습 붕괴 지표](assets/rl_v2_goal_sac_collapse_20261003.png)

[그림 원본 지표와 물리 실행 출처](assets/rl_v2_goal_sac_collapse_20261003.json).
거리 그래프는 **종료 시 더 먼 손의 거리**이며 에피소드 평균 거리가 아니다.

## 원인 후보와 확인된 사실

GPU3은13,190–18,774 actor update 사이 처음9/9 성공했다. Prior 손실 가중치는
3.41에서0.61로 떨어졌다. 다음 실행이20,384 update에 도달하면서 가중치가0이
되었고, 그 이후39개 훈련은 모두 실패했다. 왼손 거리가16.7cm로 벌어지고 이후
랙 충돌이 반복됐다. 실패 중에도 Q 추정값이 상승했으므로 TD loss 감소만으로
실제 파지 성능이 개선됐다고 판단할 수 없다.

기존에는 실제 데모 replay 혼합과 frozen 신경망 prior 손실을 **같은 schedule로
동시에 제거**했다. 성공 동작 주변에서 측정한 좁은 replay로 학습한 Q gradient가
그 영역 밖으로 actor를 이동시키는 것이 주요 원인 후보다. 감쇠와 붕괴의 시간적
일치는 뚜렷하지만 단독 인과를 확정한 실험은 아니다. Gamma0.999 수정만으로도
실패했으므로 할인율 불일치 하나로 모든 실패를 설명하지 않는다.

## 수정한 학습

`PoseGoalSACPilot`의 새 BC 초기화와 명시적인 checkpoint fork에 다음을 연결했다.

- 실제 데모 replay는 기존처럼20%에서20,000 actor update 동안0%로 감소한다.
- Frozen **신경망**의 현재 관측·clock 출력과의 actor MSE 가중치 floor10을 유지한다.
  실행 중 데모 명령 배열을 공급하지 않으며, 이 손실은 가상의 Q 전이를 만들지 않는다.
- SAC가 생성하는 normalized absolute goal을 neural prior 출력의±0.05 범위로 제한한다.
  실제 범위는 goal scale에 비례한다. 이번 초기 모델은 base XY 약±2.33/2.71mm,
  yaw±0.00840rad, 팔 관절 약±0.015–0.058rad이다. 환경의 관절/충돌 기준은 그대로다.
- 과거에 **실제로 실행한** off-policy action과 reward는 clipping하거나 수정하지 않는다.
  Q 데이터에는 이전 실제 명령을 그대로 보존하고, inverse goal decode 재현 검사를 유지한다.
- 몇 회 학습마다 동일한 개발 배치에서 optimizer를 끈 평가를 한다. 성공 수가 검증된
  최고 actor보다 줄면 그 actor만 복구한다. 최신 critic·critic optimizer·실제 성공/실패
  replay·전체 update count는 유지한다. Actor optimizer moments는 초기화한다.
- 개발 배치는 선택과 진단에 반복 사용하므로 독립 최종 성능으로 보고하지 않는다.
  마지막 평가 seed600–611은 앞선 튜닝에서 쓰지 않은 별도 배치다.

코드: `src/kuavo_isaaclab_scene/rl/multi_box/experiments/pose_goal_sac.py`,
`scripts/rl/fork_pose_goal_sac.py`, `scripts/rl/recover_pose_goal_actor.py`,
`scripts/rl/layout_residual_with_drive.py`.

## 실행과 지속 작업

GPU3의 `goal_guarded_base_box_sac_gpu3_20261003_160950` 실행을10/03 16:11 KST에 시작했다.
Parent와 Isaac child 모두 `CUDA_VISIBLE_DEVICES=3`; 내부 장치는`cuda:0`이다.
새 checkpoint는 기존 성공 actor12,494와 실제 replay7,192개에서 fork했다.
Gamma0.999에 맞추는 critic-only warmup2,000회 후 actor12,494/critic14,994로 시작한다.
실행 전 실제 native seed410개와 모든 goal/replay contract를 확인했다.

24개 랜덤 훈련 배치×2회, 처음 개발 평가4회와 훈련4회마다 개발 평가4회,
마지막 새 holdout12회로 총112개 실제 실행을 예정했다. 10/03 17:24 KST 기준,
학습 전 개발 기준 평가는 **4/4 성공**, 이어진 실제 SAC 훈련은 **4/4 성공**이다.
Actor는15,310까지 갱신됐고 첫 개발 재평가도3/3 성공 후 네 번째 배치를 수행한다.
기준 평가에는 기존 BC가 시간 초과한405/406도 포함한다. 종료된 실행의 Drive 검증은
완료됐다. 과거 붕괴 지점20,000 update 이후와 새 최종 holdout은 아직 확인 전이며,
장기 개선이나 독립 최종 성능으로 해석하지 않는다. 현재는 정확한 복구 진단을
위한 sequential one-env이며80GB VRAM을 채우는 병렬 학습은 아니다.

박스는 동적 물체이며 안쪽2–4cm·yaw±1°·주변 박스0–3개 randomization을 유지한다.
실제 초기 base는 rack-frame 좌우±20cm·밖으로3–25cm·yaw±15°다. Curriculum,
성공 판정, rack10N/obstacle5N, self-collision off 설정을 변경하지 않았다.
기존 Drive remote를 재사용해5분마다 백업하고, 검증된 오래된 checkpoint만 정리하며
로컬에 최신2개와 형식별 검증된 최신2개를 보호한다. 복구 replay도 업로드·검증한다.

사용자의 성공까지 계속하라는 요청에 따라 Codex의 **지속 목표를 active로 등록**했다.
이 도구는 현재 thread에서 후속 작업을 자동으로 이어가는 기능이다. 현재 연결에는
`automation_update` 예약 도구가 없어 특정 시각의 채팅 예약을 만들었다고 주장하지 않는다.
실행 종료·오류·성능 퇴행을 확인하고 분석·수정·재학습을 계속하며, 목표를 아직 완료로
표시하지 않는다. 다른 사용자 GPU1/2 실행과 파일에는 변경하지 않았다.

## 위 선반과 남은 검증

위 선반의 current-rest IK 진단은900tick 시간 초과였고, 이어진 base6cm 접근 진단은
566tick에`zarm_l4_link`–rack35.51N으로 실패했다. 모두 종료·백업 검증됐다.
따라서 위 선반은 여전히 미해결이다. Base 접근량을 늘리는 것만으로 해결된다고
보지 않으며, 팔 진입 자세와 그리퍼 닫힘 시점을 함께 검토해야 한다. GPU0에서
`upper_coordinated_reach_gpu0_20261003_161948` 진단을 시작했다. 실제 생성된
폴더와 상태로 완료 여부를 확인한다. Base 접근3cm, 양손12mm 안에 들어온 뒤
함께 닫기, current-rest IK와 torso 앞쪽4cm를 비교했다. 이 실행도578tick에 왼팔
`zarm_l4_link`–rack44.41N으로 실패했다. Gripper는 열려 있었고 왼손 목표에2.83cm
IK reach projection이 남았다. 종료·최종 Drive 검증 완료. 접근 거리를 줄이고
닫힘을 늦추는 것만으로 도달 범위·팔 진입 문제를 해결하지 못했다. 이 VR/IK 실행은
SAC가 아니며 실패 데이터에서 가상의 native 성공 seed를 만들지 않는다.

이번 정책 제약·legacy resume·실제 replay 보존·개발 평가 rollback 회귀 검사
**73 passed**, 기존 SAC·replay·Drive·배치·영상 회귀까지 포함해 **130 passed**.
새 제약 적용 후의 파지 성공률은 실제 실행 결과로 추가 보고한다.

## 위 선반: 높이 상한, IK projection과 닫힘 gate (10/03 후속)

실제 torso XZ는 약(0.0629,0.7129)m까지 올라간다. 현재 upright height profile의
최고 Z는 nominal0.3127m+height0.40m=0.7127m이므로 이미 높이 상한이다.
이 상태에서 동일 profile의 목표만 더 위로 지정해도 clamp되어 도움이 되지 않는다.
이번 진단에서는 torso pitch·travel profile과 기존 SAC 물리 계약을 변경하지 않았다.

URDF IK의 gross reach sphere가 전체 길이의0.95를 사용해 왼손 요청 목표를
2.826cm 잘라내는 것을 확인했다. 목표의 어깨 기준 거리는0.756034m로 현재 tool
offset을 포함한 arm reach0.766076m의0.9869다. 실제 관절 범위 안에0.9983 reach의
FK pose도 존재한다. Offline bounded IK는1.0 sphere에서 요청 목표에 약1.2e-10m
오차로 도달했고 관절 여유0.212rad를 유지했다. **이는 운동학 결과이지 실제 파지나
랙을 피하는 경로의 증명이 아니다.**

`UrdfArm(..., reach_fraction=...)`와 VR 진단 전용
`--vr-arm-reach-fraction`을 추가했다. 기본0.95는 그대로이며 진단의1.0도 기존
관절·속도·가속도 범위를 지킨다. SAC/BC/recorded action과 함께 지정하면 거부한다.

| 실제 GPU0 VR/IK 진단 | 결과 | 종료 순간 rack force / body |
|---|---|---|
| sphere0.95, base3cm, 양손12mm 동시 닫기 | 578tick unsafe, pinch0 | 44.41N / zarm_l4_link |
| sphere1.0, base 추가 접근0, 양손12mm 동시 닫기 | 608tick unsafe, pinch0 | 12.98N / zarm_l7_link |
| sphere1.0, base 추가 접근0, 손별12mm 닫기 | 616tick unsafe, pinch0 | 16.00N / zarm_l7_link |

모두 current-rest·closing-axis IK·torso 앞쪽4cm이며 종료·Drive 크기/MD5 검증 완료다.
첫 실험에는 base3cm 보조도 있어서 force 감소를 reach 비율 하나의 효과로 단정하지
않는다. Full reach여도 실제 박스가 움직이면서 왼손 목표 오차가 남고 손목이 랙에
닿는다. 오른손은 flap 최근접 거리0까지 왔지만12mm **TCP 목표** gate에서 닫기
명령이4tick뿐이었다. 목표 오차와 flap 최근접 거리는 서로 다른 값이다.

![위 선반 실제 접근·닫힘·충돌 측정](assets/rl_v2_upper_reach_gate_20261003.png)

[그림 원본 측정과 run 경로](assets/rl_v2_upper_reach_gate_20261003.json).
[실제 실패 영상](assets/rl_v2_upper_full_reach_failure_20261003.mp4)은 GPU0 PhysX 자세를
CPU mesh로 표시한 VR/IK 진단이다. SAC 성공 영상이 아니다. H.264/avc1·yuv420p·
faststart로 저장했고 전체 decode를 검증했다. Notion의 요약·상세 기록에도 native
영상과 그림으로 첨부했다.

`upper_staggered25_full_reach_gpu0_20261003_172051`의 손별25mm gate 진단은
900tick 시간 초과, unsafe/invalid0으로 끝났다. 오른손의 실제 pinch가63tick(2.1초)
검출됐고 왼손은0이다. 오른손 pinch는717–790tick 사이에 있었으나 그 전 구간
연속 유지가 아니므로2.47초 연속 grasp로 해석하지 않는다. Contact phase에서
닫기 명령은 왼손39tick·오른손317tick이다. 왼손 최소 flap 거리는9.87mm였고
TCP 목표 오차는최소20.10mm다. 마지막 왼손 gross reach projection은33.99mm로,
박스가 이동하며 다시 도달 범위를 벗어났다. 종료·최종 Drive 크기/MD5 검증 완료.

후속 `upper_staggered25_base3_gpu0_20261003_173321`은 같은25mm gate에 bounded
base3cm 접근을 더했지만579tick에`zarm_l4_link`–rack24.26N으로 실패했다.
`upper_staggered25_torso8_gpu0_20261003_174231`은 base 추가 접근 없이 기존 travel
profile 안에서 torso 전진을4→8cm로 바꿨지만613tick에 같은 왼팔–rack47.45N으로
실패했다. 두 진단 모두 양손 pinch0, 종료·최종 Drive 검증 완료다. 접근량을 키워
왼손 reach sphere projection을 없애도 실제 팔 경로가 랙에 닿을 수 있다.

`upper_staggered25_full_wrist_gpu0_20261003_175310` 진단은1.0 reach·torso4cm·
손별25mm를 유지하고 성공 demo의 full wrist orientation을 맞췄다.900tick
시간 초과, unsafe/invalid0, 양손 pinch0으로 종료·Drive 검증 완료다.
마지막 flap 거리는 왼손20.73mm·오른손7.24mm였다. 전체 손목 자세를 맞추는
것만으로 남은 도달·파지 문제를 해결하지 못했다.
양손 opposing flap·8mm clearance·0.25초 유지의 최종 성공 조건,
rack10N/obstacle5N은 그대로다. 위 선반 actual positive native seed는 아직 미확보다.

## 여러 실제 성공 episode를 학습에 연결

현재 SAC의 첫 두 훈련 성공으로부터 **830개 실제 전이(416+414)**를 읽었다.
현재 물리/reward 계약, GPU 실행, 초기 제어 상태, step 시간과 관측 연속성,
최종 양손 pinch·성공 관측을 검사했다. Physical action inverse의 최대 오차는
1.073e-6이며 action/reward를 수정하지 않았다. 해당 pool은 현재 실행에 주입하지
않았고 `actual_multi_seed_import_audit.json`에 출처·검사 결과를 저장했다.

- `read_executed_successes`는 같은 strict 물리 검사로 현재 goal-SAC/BC 등의 실제
  성공을 받는다. Collection source만으로 성공을 인정하지 않는다.
- `--native-dataset`과 `--pose-student-native-seed`를 반복 지정할 수 있다.
  Episode마다 clock0과 각 초기 박스 anchor를 따로 사용하며 terminal 경계를 넘지 않는다.
- 단일 파일의 기존 goal contract/hash는 유지한다. 서로 다른 성공 episode를
  결합한 계약은 episode context와 ordered source SHA256을 명시한다.
- 원래410개 단일 데모로 만든 목표 범위를 벗어나는 실제 inverse command가
  두 실행에서 각각21/24행 있었고 normalized 최대값은2.398/2.418이다. 초기 base
  차이와 servo의 physical clipping 때문에 실제로 같은 명령을 재현하는 inverse
  goal이 옛 목표 범위 밖에 있을 수 있다. **실제 action을 clipping해서 억지로
  넣지 않는다.** 새로운 pool의 실제 goal label로 center/scale을 다시 fit하고,
  새 정책을 물리에서 검증한 뒤 별도 SAC 계약으로 시작해야 한다.
- 위/아래 선반을 묶을 때는 full observation fit을 사용한다. Clock-only로 서로
  다른 동작을 같은 시간 입력에 학습하지 않는다. 위 선반 실제 성공이 아직 없어
  해당 positive 데이터는 미확보다.

```bash
# 파일은 현재 물리에서 성공한 실제 GPU 전이여야 한다. Output은 새 고유 경로다.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/fit_v2_pose_student.py \
  --native-dataset /absolute/path/to/actual-success-a/executed_transitions.hdf5 \
  --native-dataset /absolute/path/to/actual-success-b/executed_transitions.hdf5 \
  --initial-box-relative --clock-horizon 900 --steps 20000 \
  --output-dir /absolute/path/to/unique-pooled-student
```

새 pose/seed/URDF 진단과 SAC·배치·Drive 관리 회귀 검사 **153 passed**.
학습과 성공 판단은 실제 완료된 물리 결과로 계속 확인한다.

Demo TCP frame이 현재 calibrated closed TCP와 다른지도 따로 확인했다.
데모2개와 현재 물리 진단에서30tick마다 기록된20관절로 URDF FK를 계산했다.
두 데모의 TCP 차이는 최대0.84µm, 현재 진단은1.20µm였다. 기록된 EEF 회전도
FK와 약1.8e-6 이하 차이로 일치했다. **이 샘플 검사에서는 오래된 TCP offset
불일치가 원인이라는 가설을 지지하지 않는다.** Box/flap frame과 demo grasp offset의
동일 pinch-frame 재구성도 오차0으로 맞았다. 이와 별개로 upright torso 변경 후의
팔 경로와 live servo 추종 문제는 남아 있다.
[TCP frame 검사 방법·원본 SHA256·측정값](assets/rl_v2_demo_tcp_fk_audit_20261003.json).

## Pooled 목표 범위와 관측 기반 prior 준비 (10/03 17:53)

`pooled_lower_goal_bc_cpu_20261003_174452`는 종료·검증된 SAC **훈련 성공5개,
2,073전이**만 사용했다. 개발/최종 평가 episode를 fit에 포함하지 않았다.
CPU 한 스레드·`CUDA_VISIBLE_DEVICES=''`로20,000회 BC fit, clock horizon900,
full observation과 initial-box-relative goals를 사용했다. 기존 GPU3 학습에는
교체하거나 replay를 주입하지 않았다.

새 center/scale에서 실제5개 episode를 `PoseGoalSACPilot(training=False)`에
넣어 **2,073전이 모두** 받아지는 것을 확인했다. Episode 시작 clock은 다섯 개
모두0이고 각 초기 박스 anchor를 따로 유지한다. Normalized actual goal 최대1.0,
physical action inverse 최대 오차1.073e-6; 과거 action/reward를 clipping하지 않았다.
Actor/critic 업데이트0으로 수행한 데이터 계약 검사이며 **새 정책의 물리 성공을
뜻하지 않는다.** Fit 모델과 종료 로그의 Drive 크기/MD5 검증은 완료됐다.
새 데이터 계약 검사 기록도 별도 고유 폴더로 Drive 업로드·MD5 검증했다.

Fit 오차는 팔 관절 최대 평균0.00360rad·최대0.02457rad, base XY 평균0.256/0.403mm·
최대2.89/3.11mm였다. Offline 오차가 작다는 것만으로 실행 안정성을 주장하지 않는다.
실제 모델은 frozen 개발 평가를 통과해야 이후 SAC 초기화 후보로 사용할 수 있다.
`pooled_lower_bc_dev406_gpu0`에서 새로운 모델의 optimizer를 끄고 반복 사용한
개발 배치406을 평가한다. SAC 실행이나 새 독립 최종 holdout 성공으로 기록하지
않으며, 해당 평가 전이를 training seed로 사용하지 않는다. Native reader는
`layout.split=validation/holdout/eval/evaluation` 데이터를 training seed에서 거부한다.
평가 데이터 누출 거부 회귀를 추가한 관련 검사 **157 passed**.

같은 시각 기존 GPU3 SAC는 **훈련7/7 성공**, 학습 전 개발4/4·첫 개발 재평가4/4,
actor17,410으로 다음 훈련을 수행한다. 완료된15개 실행의 Drive 검증은 끝났다.
아직20,000update 이후와 새 최종 holdout600–611은 확인 전이다.
