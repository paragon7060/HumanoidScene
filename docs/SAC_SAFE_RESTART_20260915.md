# 2026-09-15 박스 안전 종료 적용 SAC 새 학습

GPU 3만 `CUDA_VISIBLE_DEVICES=3`으로 사용한다. 이전 checkpoint의 오염된
normalizer/optimizer를 복원하지 않는 새 학습이다. 최신 main eba8e0b에 로컬
미커밋 코드 변경을 합친 source snapshot을 사용하고 SHA256을 기록했다.
원래 작업 폴더와 다른 사용자의 프로세스/파일, 50GB 예약 파일은 변경하지 않았다.

- 전체 관절 25 action, policy 관측 269차원, 오른손 flap 파지 후 6cm lift / 0.5초 유지.
- 4096 env, GPU replay 3000만 transition, rollout 16, batch 1024, updates 4/vector step.
- 유효 transition 200만 개와 최소 450 vector step을 충족한 후 업데이트.
- 1500 iteration, 50 iteration마다 및 마지막 checkpoint 저장, 실행 상한 24시간.
- GPU 프로세스 사용 한도 77824 MiB, 전체 GPU 여유 2048 MiB, 로컬 여유 5 GiB.
- box 50cm 과도한 lift / 비유한 pose·velocity / 선속도 10m/s·각속도 100rad/s 초과 실패.
- 비정상 box 종료 step shaping 0, 기존 failure 비용 −60 유지.
- 비유한 관측/transition normalizer·replay 제외; settling 데이터 제외.
- Curriculum 없음. 현재 승인된 충돌 제한 OFF, 공통 그리퍼 마찰 20/16 유지.
- 로컬에 추가된 정렬·닫기 보상 함수는 기존 reward manager에 연결하지 않았다.
  현재 활성 보상 설정을 사용하며 PPO 정책 변경은 SAC에 사용하지 않는다.

## 보관과 현재 제한

기존 Drive 업로더가 300초마다 크기·MD5를 검증하고 최신 2개 및 최신 검증 2개를
보호한다. 학습기 자체 삭제는 비활성이다. 종료 후 닫힌 로그도 검증·업로드한다.
**시작 시 기존 Drive 인증이 invalid_grant로 실패했다.** 새 인증은 만들지 않았다.
업로더는 자동 재시도하지만 인증 복구 전에는 실제 업로드를 보장하지 않으며,
미검증 checkpoint는 로컬에 모두 보존한다. 최초 로컬 여유는 약 322 GiB였다.

관리 폴더: `artifacts/rl/drive_runs/sac_gpu3_all_joints_safe_20260915_151947_b1137977`.
실행 폴더: `artifacts/rl/drive_runs/sac_gpu3_all_joints_safe_20260915_151947_b1137977/sac_20260915_152035_9b145b`.
소스 스냅샷: `artifacts/rl/source_snapshots/sac_all_joints_safe_20260915_151947_b1137977`.
원격 대상: `gdrive:HumanoidScene-RL/sac_20260915_152035_9b145b`.

실행 상태는 관리 폴더 `status.json`, 진행률은 실행 폴더 `metrics.jsonl`을 읽는다.
기록만으로 실행 중 또는 업로드 완료로 판단하지 않는다. 재시작 전 필수 CPU
테스트 18개가 통과했고, 실제 4096 env 데이터 수집 시작을 확인했다.

## 실제 업데이트 시작 확인

2026-09-15T15:27:41.804693+09:00: iteration 35, 유효 transition 2,170,787, optimizer update 167. Q loss 0.03669, actor loss -3.83881, alpha 0.09511가 모두 유한했다. 비유한 데이터 제외 수 0. GPU 사용 74117 MiB (약 72.38 GiB). 학습은 계속 진행한다. Drive 인증 오류는 미해결이며 업로드 검증 완료로 간주하지 않는다.

## Drive 인증 복구

2026-09-15T15:30:37.403759+09:00: 재인증 후 Drive about 조회가 정상 응답했다. 실행 폴더 manifest.json, env.yaml, agent.yaml을 실제 업로드하고 크기·MD5를 검증했다. 기존 관리자와 잠금을 재사용했으며 추가 상시 업로더는 만들지 않았다. 체크포인트 검증은 저장 후 업로드 가능한 시점에 진행한다. 학습 및 기존 300초 자동 업로드·검증 보관 정책을 계속 유지한다.

## GPU 3 checkpoint 900 재개

2026-09-15T20:51:23.713344+09:00: 사용자의 재개 요청에 따라 마지막 저장 checkpoint 900에서 600 iteration을 추가해 총 1500까지 진행한다. 중단 직전 901~916의 미저장 모델 상태는 복원하지 못한다. 모델·normalizer·optimizer 91개 텐서의 유한성 및 원본 소스 SHA256 일치를 확인했다. Replay는 checkpoint에 포함되지 않아 새로 수집하고 유효 transition 200만 개 warmup을 반복한다. GPU 3 단독·4096 env·GPU replay 3000만·50 iteration 저장·300초 Drive 검증/보관 설정을 유지한다. 기존 run과 다른 사용자의 프로세스는 건드리지 않았다.

관리 폴더: `artifacts/rl/drive_runs/sac_gpu3_all_joints_resume900_20260915_205123_25c6e50f`.
실행 폴더: `artifacts/rl/drive_runs/sac_gpu3_all_joints_resume900_20260915_205123_25c6e50f/sac_20260915_205130_c7fe3c`.
Drive 대상: `gdrive:HumanoidScene-RL/sac_20260915_205130_c7fe3c`.
현재 상태는 관리 폴더 `status.json`, 진행률은 실행 폴더 `metrics.jsonl`을 확인한다.

2026-09-15T20:53:42.764039+09:00: 실제 모델·optimizer 복원 로그와 iteration 905 데이터 수집을 확인했다. 비유한 transition 0, GPU 사용 74063 MiB. 기존 관리자가 새 run 메타데이터를 자동 업로드·MD5 검증했고 backup_error는 null이다. Replay warmup 후 SAC 업데이트를 계속 진행한다.
