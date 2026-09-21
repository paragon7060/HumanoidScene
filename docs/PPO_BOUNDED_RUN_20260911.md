# 2026-09-11 PPO 정책 수정 및 실행

기준 원격은 `f93e9e9`다. 사용자가 현재 환경·보상으로 집기/상승 학습을 요청하여
충돌 OFF/20N, 오른팔 제어, flap 전체 표면, 파지/상승 진척 보상을 그대로 사용한다.
과거 checkpoint와 호환되지 않는 새 PPO 정책으로 학습한다.

정책 구현: `src/kuavo_isaaclab_scene/rl/agents/flap_ppo.py`.
양쪽 `configs/rl_pick_arms_only.py`에서 `configure_flap_ppo()`를 호출한다.
일반 PPO 기본 설정과 SAC/DPPO 알고리즘은 변경하지 않았다.

- actor/critic running normalization의 표준편차 하한 0.1, 최종 입력 ±5.
  드문 0/1 및 파지 이력 관측의 과도한 증폭을 제한한다.
- tanh Gaussian action. likelihood에 역변환과 Jacobian을 포함하고 실제 변환된
  분포의 entropy를 재매개화 표본으로 추정한다. 실행 시 clip만 추가한 방식이 아니다.
- latent mean은 `3*tanh(network_output/3)`로 제한한다. RSL-RL의 KL 계산에는
  latent Gaussian mean/std를 전달하고, inference는 tanh(mean)를 반환한다.
- 학습 가능한 관절별 latent std는 sigmoid로 0.05~0.35에 제한한다. 초기 0.15.
  entropy coefficient는 0.005→0.001. 작은 출력층 초기화로 초기 열기 편향을 피한다.
- MLP 256/256/128은 유지. rollout 32→64 step, minibatch 4→32,
  epoch 5→4. 한 번의 수집 후 optimizer update는 128회다.
- rollout에서 최대 4096개를 추출해 KL, PPO ratio clip 비율, action |a|>0.95 비율,
  latent std min/max, critic explained variance를 기록한다. TensorBoard Loss 그룹에
  출력되는 진단값이며 loss 목적함수에 추가되는 항이 아니다.

이는 여러 정책 수정을 함께 적용한 실험이며 각 수정의 기여를 분리한 ablation은 아니다.
환경 보상의 반복 획득 가능성은 이전 재검토 기록에 남긴 상태다.

CPU 검증: 실제 RSL-RL likelihood 대조, gradient, PPO update, checkpoint roundtrip,
희귀 flag 범위 및 Drive supervisor/backup 검사 **22개 통과**.
공유 Conda 패키지는 변경하지 않았다. 기존 env_isaaclab_232를 사용한다.

GPU 1은 시작 전 비어 있었고 A100 80GB의 free 81153 MiB를 확인했다.
설정된 Drive remote와 로컬 디스크 여유 공간을 확인했다.
관리자는 CUDA_VISIBLE_DEVICES=1, 내부 cuda:0, Kit renderer 물리 GPU 1로 실행한다.
GPU 자체 사용 78000 MiB 상한과 GPU free 2048 MiB, 로컬 free 5GiB를 감시하며,
한도 초과 시 이 관리자가 생성한 자식 프로세스 그룹에만 종료 신호를 보낸다.
PPO의 완료/실패 상태 파일을 추가해 Kit가 실패를 exit code 0으로 가려도 구분한다.

Drive 업로드 주기는 300초. checksum 검증 완료된 오래된 checkpoint만 정리하고
각 형식의 최신 2개 및 검증된 최신 2개를 보호한다. 학습 종료 후 닫힌 로그도 자동
업로드·검증한다. 새 인증을 만들지 않으며 다른 사용자 프로세스는 변경하지 않는다.

검증 실행 관리 폴더:
`artifacts/rl/drive_runs/ppo_bounded_smoke_gpu1_20260911_180753_a0d207`.
실제 진행/종료 여부는 해당 status.json과 PID를 확인한다.

## 검증 결과 및 본 학습

16,384 env 검증 2회 업데이트가 정상 종료됐다. 총 2,097,152 transition이며
실측 처리량은 15,604 / 17,479 transition/s, GPU peak 28,395 MiB(27.73 GiB)다.
표본 KL 0.0133/0.0184, ratio clip fraction 0.2168/0.2903, action |a|>0.95 비율 0,
latent std 약 0.1481~0.1501이었다. 마지막 checkpoint/optimizer의 tensor 76개가 모두 유한하다.
이 검증은 학습 구동 확인이며 파지 성공이나 정책 성능 검증은 아니다.

본 학습은 기존 요청의 16,384 env를 유지한다. 80GB는 사용 가능한 상한이며
VRAM을 채우기 위해 rollout batch를 더 늘리지는 않았다. 검증 모델 model_1.pt에서
2,000 iteration을 이어서 실행한다. RSL-RL의 저장 인덱스 규칙상 로그 iteration은
1~2000이다. 저장 간격 10, 최종 모델 별도 저장, Drive 업로드 확인 주기 300초다.

- 관리자 폴더: `artifacts/rl/drive_runs/ppo_bounded_gpu1_20260911_182052_764df1`
- 실제 실행 폴더: `train_20260911_182101_98e97e`
- 관리자 PID: 3084642, 학습 PID: 3084669 (실제 상태는 status.json 및 PID 재확인)
- Drive: `gdrive:HumanoidScene-RL/train_20260911_182101_98e97e`
- checkpoint 초기값 SHA256: `f17620f5d86ef9e748909d011c01b1268557839c71697202e3ff969ba273a6c0`
- 시작 후 현재는 본 학습용 환경 초기화 단계다. 검증 프로세스 2892312는 정상 종료했고
  본 학습 프로세스만 GPU 1을 사용하는 것을 확인했다. 다른 GPU의 프로세스는 건드리지 않았다.

검증 실행은 18:22:24에 `training_exit_code=0`, `final_upload_verified=true`를 확인했다.
checkpoint 0/1, TensorBoard, console, resources 및 종료 status가 Drive에서 크기/MD5
검증을 마쳤다. 본 학습의 소스 사본·해시·검증 audit도 전용 Drive 폴더에 업로드·검증했다.
