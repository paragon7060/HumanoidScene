# V2 grasp: 데모 연결과 학습 진행 요약

2026-10-02: 훈련 배치8/8, 별도의 새 배치 평가8/8에서 양손 flap 파지 유지에
성공했다. 안전 위반·invalid reset·시간 초과는 모두0이다. 최종 actor는7,662
update이며 평가8회는 이 하나의 checkpoint를 고정했다.

기존 성공은 **데모의 기준 제어 + SAC 보정 정책**이다. 별도로 실행 중 데모
경로를 공급하지 않는 BC 목표 정책도 새 배치1개에서 성공했다. 이를 시작점으로
목표 자세 SAC를 연결했다. 독립 SAC의 성공이나 모든 선반·박스 크기·초기
자세에 대한 성공을 확인한 결과는 아니다.

## 데모와 SAC의 연결

![데모 기준 제어와 SAC 보정의 연결](assets/rl_v2_demo_sac_pipeline_20261002.png)

1. VR 성공 데모2개 중 아래 선반 데모를 현재 S63·Leju·upright torso 제어로
   재현해 실제 성공 transition410개를 확보했다. 위 선반 데모는 현재 제어에서
   아직 성공하지 못했다.
2. 기준 경로의 관절·base·torso 목표를 인식된 박스 위치에 맞춰 변환한다.
   경로와 그리퍼 타이밍은 실행 중에도 제공된다.
3. SAC는 현재 로봇·랙·모든 박스 관측과 기준 목표를 보고22차원 보정을 낸다.
   팔 목표의 최대 보정은±0.03rad이다. 기준 제어와 합쳐 실제24차원 명령을 낸다.
4. 실제 물리 실행의 관측·보상·종료 결과만 replay에 넣어 actor/critic을
   학습한다. 데모의 제안 동작을 가짜 성공 전이로 사용하지 않는다.

데모에서 확보한 실제410개 전이는 zero-residual 초기 데이터다. Actor의
zero-residual 초기화 prior는512 updates 동안 사라지지만 기준 제어는 남는다.
모방 초기화가 끝났다는 것과 데모 경로 의존성이 없다는 것은 다르다.

## 측정 결과

| 실험 | 결과 | 해석 |
| --- | --- | --- |
| 이전 일반24-action SAC actor350 | 파지0, 시간 초과 | 독립 정책은 아직 실패 |
| 데모 기준 경로 재현 | 아래 선반 성공 / 위 선반 실패 | 학습 없이 가능한 기준 동작 점검 |
| 다양한 배치 SAC 훈련8회 | 8/8 성공, actor7,662 updates | 실제 보정 학습 중 결과 |
| 같은 모델의 새 배치 평가8회 | 8/8 성공, 추가 optimizer/normalizer update0 | 해당 제한 분포에서의 고정 모델 성능 |
| 같은 배치에서 SAC 보정0 | 3/3 성공, 각410tick, Drive 검증 완료 | 기준 제어만으로도 해결됨; SAC의 개선 효과 미확인 |
| 상태 기반 절대 목표 BC 첫 모델 | 872tick 시간 초과 | Offline fit과 물리 성공은 다름 |
| 경과 시간 + 초기 박스 상대 목표 BC | 새 배치1/1 성공,412tick, SAC update0 | 실행 중 데모 경로 없는 학습 모델의 물리 성공 |
| 위 BC에서 목표 자세 SAC 연결 | 첫 훈련1/1 성공,411tick, actor696/critic1,196 | 나머지 훈련·고정 평가 진행; BC 대비 개선은 미확인 |

성공 범위는 작은 박스·아래 선반 같은 구역·안쪽2–4cm 위치 변화·주변 활성
박스0–3개다. 시작 yaw는±1°지만 롤러에서0에 가까워지므로 큰 yaw 일반화로
해석하지 않는다. 보정0에서도 성공한 세 배치 때문에8/8을 SAC의 개선 효과로
해석할 근거는 아직 부족하다. 기준 제어의 성공과 학습의 기여를 별도로 측정한다.

실험: `artifacts/rl/drive_runs/layout_suite_gpu3_20261002_0440/results.json`.
훈련/평가16회에서 관측된 actor updates와 종료 결과, 각 run의 최종 Drive 검증이
기록되어 있다. 임의의 전체 배치에 대한 통계적 보장은 아니다.

## 실제 새 배치 성공 영상

평가 배치1: 안쪽2.86cm 이동, 주변 활성 박스3개. 고정 모델이411tick(13.7초)에
양손 파지 유지에 성공했다. 실제 GPU3 PhysX 자세를 CPU mesh로 표시한다.
우측 대기 영역의 비활성 박스는 학습의 활성 주변 박스와 구분해야 한다.

[![새 배치의 양손 flap 파지 성공](assets/rl_v2_layout_heldout_success_20261002.png)](assets/rl_v2_layout_heldout_success_20261002_h264.mp4)

[H.264 영상](assets/rl_v2_layout_heldout_success_20261002_h264.mp4) ·
[실제 배치·종료·고정 모델 기록](assets/rl_v2_layout_heldout_success_20261002.json).

## 진행 중인 확대와 남은 문제

- GPU0: 보정0 비교3/3 종료·Drive 검증 완료. 새 BC 정책의 성공을 확인한 뒤
  경로 공급 없는 목표 자세 SAC를 실행한다. 아래에 데이터 연결과 action 계약을 기록한다.
- GPU3: 앞뒤±1cm 확대는4개 성공 후5번째 초기화에서 footprint 경계 이탈로
  중단됐다. 잘못된 장면의 전이를 Q에 넣지 않았다. 첫 배치의 초기 depth
  +5.08mm도 정착 후0.0002mm가 되어 앞뒤 일반화의 증거가 아니다.
  마지막 정상 actor10,446에서 새 훈련12개·평가12개를 재개했다. 분포는
  **안쪽2–3.5cm, 추가 yaw/depth0, 주변 박스0–3개**다. 실제 사용되는 원본
  데모 초기 자세로24개 모두 기존2mm footprint margin을 검사했다. Reset의
  기존 shelf/region 판정은 유지한다. 첫2개가 성공했으며 전체 평가율은 아직 없다.
- 위 선반: full-wrist/closing-axis IK, torso 앞쪽4cm, 양손 동시 닫기,
  원본 gripper timing을실제로 점검했다. 각각과 일부 조합이 모두900tick 시간 초과였다.
  초기 flap 상태 등 구 데모에서 복원할 수 없는 상태도 있어 완전한 원본 재생은
  아니다. 좌표계·접촉·관절 제한·시점을 계속 분리해서 확인해야 한다.
- 기준 경로 의존성을 줄이는 독립 정책은 별도로 확인해야 한다. 위 선반도
  실제 성공 데이터를 확보한 후 학습에 연결하며, 실패를 해결되었다고 보지 않는다.

현재 reward, 양손 flap 성공 조건, 랙10N/주변 장애물5N, self-collision 제외와
upright torso는 유지한다. Curriculum을 추가하지 않았다. GPU3 재개 실행은
`layout_stable_gpu3_20261002_0720`, 보정0 비교는
`layout_zero_baseline_gpu0_20261002_0617`이다.

## 경로 공급 없는 BC 목표 정책과 SAC 연결

첫 상태 기반 BC 모델은 실제 훈련 성공3,292개 명령을 모방했지만 새 배치에서
시간 초과였다. 낮은 offline loss만으로 물리 성공을 판단하지 않았다. 다음 모델은
관측 입력 가중치를 초기 BC 동안0으로 두고 경과 시간에 따른 목표를 학습했다.
Base 목표 XY는 **초기 인식 박스 위치에 대한 상대값**으로 표현해 새 박스
위치로 이동하도록 했다. 나머지는17개 관절, torso X/Z, base yaw, 양 그리퍼다.
초기 박스 anchor와 현재 로봇 상태를 사용하는 servo는 기존 물리 제어기다.
실행 중 경로 배열·목표 배열·원본 명령을 제공하지 않는다.

현재 물리에서 재현한 아래 선반 성공410개 명령으로 BC20,000 updates를 수행한
모델이 fit에 사용하지 않은 배치에서 **412tick(13.73초)에 양손 held grasp**에
성공했다. Unsafe/invalid/timeout0이며 마지막 실제 pinch는[true,true]였다.
시작 장면 복원에는 데모 초기 자세를 사용한다. 이 성공은 BC이며 SAC update0이다.
시간 기반의 제한된 정책으로, 임의 시작 자세·속도·박스 크기 일반화는 미확인이다.

[![데모 경로 공급 없는 BC 정책의 양손 파지](assets/rl_v2_bc_no_live_reference_success_20261002.png)](assets/rl_v2_bc_no_live_reference_success_20261002_h264.mp4)

[실제 BC 성공 기록](assets/rl_v2_bc_no_live_reference_success_20261002.json).

SAC 연결은 별도 `pose_goal_sac.py`가 관리한다. Actor441/critic533은 현재 관측,
모든 박스, 경과 시간과 초기 박스 anchor를 포함한다. Action24개는 정규화된
목표 자세 좌표다. 원래 delta-action24개나 residual22개와 Q/replay를 섞지 않는다.
실제410개 전이의 action을 정확한 역변환으로 목표로 바꾸고, decode와 닫힘
gate를 적용해 **실제로 실행했던 명령과1e-5 이내로 일치**해야 Q seed로 허용한다.
현재 reward/terminal 관측과 성공 계약은 기존 검사를 그대로 통과해야 한다.

Critic500회 warmup 후 첫64개 실제 전이를 모으고 SAC actor/critic을 step당2회
갱신한다. Actor LR2e-6, std cap0.02, entropy backup 제외, Q 정규화와 critic
LayerNorm을 사용한다. 초기 데모 전이20%와 frozen BC 신경망의 actor-only prior는
4,000 actor updates 동안0으로 줄인다. 이 prior는 실행 중 경로를 조회하지 않고
Q의 가짜 전이도 만들지 않는다. 초기 BC에서0이던 관측 가중치도 SAC는 학습한다.

실행 `pose_goal_sac_gpu0_20261002_0735`는 훈련3개 후 별도 평가3개에서 같은
최종 모델과 normalizer를 고정한다. 실제 성공률은 완료된 metrics로 확인해야 한다.
첫 훈련 배치에서는 실제411개 전이를 수집하며 actor696/critic1,196회를 갱신하고
양손 held grasp에 성공했다. Unsafe/invalid/timeout0, 최종 Drive 검증 완료다.
실행 중 데모 경로를 공급하지 않는 SAC 모델의 **훈련1회 성공**이며 고정 평가나
BC 대비 개선 효과를 입증한 결과는 아니다.
두 번째 훈련 배치도 성공했고 actor1,384까지 이어졌다. 고정 평가 결과는 아직 없다.

[![목표 자세 SAC의 훈련 중 양손 파지](assets/rl_v2_pose_goal_sac_training_success_20261002.png)](assets/rl_v2_pose_goal_sac_training_success_20261002_h264.mp4)

[실제 SAC 성공 기록·학습 횟수](assets/rl_v2_pose_goal_sac_training_success_20261002.json).

Checkpoint512 actor updates 간격과 종료 시 저장, 기존5분 Drive 업로드·검증
후 보존 규칙을 그대로 사용한다. 모든 Isaac 자식에 CUDA_VISIBLE_DEVICES를 지정한다.
첫 goal trial은 업로더가 요구하는 env/agent 메타파일이 없어 종료 후 업로드만
수행했다. 메타파일 저장을 보강했고, 이미 실행 중이던 두 번째 trial에도 실제
계약을 기록해 주기 업로드를 활성화했다. 종료한 첫 trial의 backup은 검증됐다.

```bash
PYTHONPATH=src:scripts/rl python scripts/rl/fit_v2_pose_student.py \
  --native-dataset /absolute/path/to/current_success/executed_transitions.hdf5 \
  --initial-box-relative --clock-only-fit --steps 20000 \
  --output-dir /absolute/path/to/unique-bc-fit

PYTHONPATH=src:scripts/rl CUDA_VISIBLE_DEVICES=0 python scripts/rl/layout_residual_with_drive.py \
  --policy-mode pose-goal --gpu 0 --train-count 3 --eval-count 3 \
  --experiment-dir /absolute/path/to/unique-goal-sac-run \
  --layout-dir /absolute/path/to/preflight-checked-layouts \
  --checkpoint /absolute/path/to/unique-bc-fit/student.pt \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 --episode-index 0 \
  --training-manifest /absolute/path/to/current-v2-run/manifest.json \
  --pose-student-native-seed /absolute/path/to/current_success/executed_transitions.hdf5 \
  --steps 900 --capture-every 30
```

GPU 번호를 변경할 때 CUDA_VISIBLE_DEVICES와 --gpu를 같이 변경한다. 이 경로는
일반 SAC runner에서 불러올 수 없는 별도 action 계약이다. 이전 residual 모드의
기본 실행은 유지된다. Optional `--time-harmonics 16`은 시간이 입력인 모델의
표현력 실험이며 현재 성공 모델은 harmonics0을 사용한다.

## 영상 형식과 보관

새로 올린7개 영상은 mp4v 코덱이었고 그중2개는 application/mp4 타입으로
일반 파일에 등록되어 있었다. H.264(avc1, yuv420p, faststart)·video/mp4로 변환한
복구본을 기존 Notion 기록에 추가했다. 원본 frame 수·FPS·해상도와 전체 decode를
확인했다. 기존 Drive 원본은 유지한다. 이전 첨부의 직접 교체가 거부되어 새 복구
영상과 아래 요약 페이지를 사용한다. 브라우저 재생 자체를 직접 확인한 것은 아니다.

새 물리 재생의 저장은 writer 종료 → H.264 변환 → 전체 decode 검사 → 원자적
파일 교체 → 완료 상태 기록 순서다. 변환 실패 시 원본을 남기고 실험을 실패로
기록한다. CPU만 사용한다. `scripts/rl/browser_video.py`의 CLI로 이미 종료된
영상도 다른 경로에 변환할 수 있다. 닫힌 원격 원본을 덮어쓰지 않는다.

이전 targeted CPU tests107개 통과. 새 목표 action 연결 후 관련 **98 passed**:
목표↔실제 명령 역변환, 닫힘 gate와 entropy mask 일치, anchor 추가 시 BC
가중치·normalizer 보존 및 기존 SAC/replay/Drive/배치/영상 검사다. 영상 코덱·timing 보존·변환 실패 시 원본
보존과 배치 depth/footprint 검사를 포함한다. 실제 성공률은 PhysX 결과로 판단한다.
Drive는 기존 인증으로5분마다 검증 업로드하며 최근2개 checkpoint 및 보호된
검증본을 유지한다. 다른 사용자의 파일·프로세스는 변경하지 않는다.

[읽기 쉬운 Notion 하위 페이지](https://app.notion.com/p/3ec63918d42a81389724c8cc53084726) ·
[자세한 일반화 보고서](RL_V2_LAYOUT_GENERALIZATION_20261002.md) ·
[Drive 보관 규칙](RL_GOOGLE_DRIVE.md).
