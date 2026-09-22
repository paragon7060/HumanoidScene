# Multi-box v2 SAC 초기 배치 반복 실패 진단

2026-09-22 14:19 KST 기준. 1664 env, s63, leju-twofinger, rack-rollers,
self-collision 비활성화, GPU3 실행을 조사했다. 별도 재현은
`CUDA_VISIBLE_DEVICES=1`로 격리한 8–64 env, zero-action 실험이다.

## 결론과 적용 결과

반복 종료의 주원인은 PhysX 접촉 patch 버퍼 부족으로 판단한다.
이전 6회 모두 초기 안정화에서 종료되었으며 SAC 업데이트에는 도달하지 못했다.
버퍼를 `2**20`에서 `2**21`로 늘린 현재 실행은 **1664개 모두 91 control step에
준비 완료**했고 첫 rollout iteration까지 통과했다. 환경 수는 줄이지 않았다.

| 지표 | 이전 6회 | 수정 후 현재 실행 |
| --- | --- | --- |
| patch capacity | 1,048,576 | 2,097,152 |
| `Patch buffer overflow` | 실행당 703건 | 첫 iteration까지 0건 |
| 초기 안정화 | 180 step 후 64–76 env 미준비, 종료 | 91 step에 1664/1664 준비 |
| 초기 배치 invalid 누계 | 4,321–4,498 | 1,708 |
| 완료 rollout iteration | 0 | 1 |
| rollout 중 invalid reset | 진입 못함 | 0 |
| 유효 transition | 0 | 53,248 |
| nonfinite transition | 진입 못함 | 0 |
| optimizer update | 0 | 0: warmup 중 |

이는 초기 진입 장애의 개선을 확인한 결과다. 장시간 안정성이나 파지 성공률 검증은 아니다.
현재 replay warmup 목표는 748,800 transition이며 성공 종료는 아직 0건이다.

## 반복 종료의 원인

### 1. 접촉 버퍼 용량 초과

이전 로그는 최소 약 1,065,258–1,065,977개 patch 용량을 요구했다.
할당된 1,048,576개보다 많다. VRAM의 남은 공간과 별도로 설정하는 고정 용량이라
VRAM 여유가 있어도 발생한다. 해당 6회의 종료 원인은 CUDA OOM이 아니라
`Initial v2 reset settling did not finish in 180 steps`였다.

롤러는 환경당 546개로, 1664 env에서는 908,544개다. 박스와 로봇의 접촉까지 있어
작은 환경 수에서 동작했던 버퍼가 큰 실행에서는 부족했다. NVIDIA의
[접촉 버퍼 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/guides/collision_guide.html)도
capacity를 초과하는 경우 접촉 계산의 정확성을 보장할 수 없으며 해당 버퍼를
늘리도록 설명한다. overflow가 사라지고 모든 환경이 안정화된 현재 결과가
이번 실행에 대한 원인 판단을 뒷받침한다.

### 2. 전체 환경을 기다리는 초기 진입 조건

기존 runner는 1664개 중 하나라도 180 step 안에 준비되지 않으면 전체 실행을
실패시켰다. 대부분은 준비됐어도 나머지 64–76개의 재배치 때문에 진행하지 못했다.
2시간 재시작은 같은 설정을 그대로 반복하여 같은 오류를 재현했다.

runner에는 제한 시간 이후 90% 이상 준비된 경우 진행하는 보완을 추가했다.
미준비 행은 기존 rollout의 replay/normalizer 마스크로 제외한다.
90% 미만이면 계속 실패 처리한다. **현재 실행은 91 step에 전부 준비되어
이 보완 경로를 사용하지 않았다.** 따라서 이번 통과를 단순한 검사 완화의 결과로
해석하면 안 된다.

## 별도로 남아 있는 최초 reset 문제

접촉 버퍼가 충분한 소규모 환경에서도 첫 박스 배치 직후 큰 이동이 발생한다.
16 env 일반 랙 실험의 대표 박스는 첫 control step(1/30초)에 약 **0.657 m** 이동했다.
reset 직후 조회한 root는 랙 위치였지만 flap 링크 위치는 바닥 대기 위치에 남아 있었다.
이는 articulation teleport 시 root와 하위 링크 상태의 동기화 문제를 의심하게 한다.
단, 조회 버퍼와 실제 solver 상태를 완전히 분리한 진단까지는 하지 않았으므로
정확한 PhysX 내부 원인을 확정한 것은 아니다.

| 재현 조건 | 관측 결과 |
| --- | --- |
| 롤러 랙, 64 env | 첫 reset 64개 invalid, 추가 1개; 40 step 시점 전부 준비, 105 step까지 유지 |
| 일반 랙, 16 env | 첫 reset 16개 invalid; 45 step 시점 전부 준비 |
| 물리 8 substep 예열, 16 env | 첫 reset 16개 invalid, 추가 1개; 이후 전부 준비 |
| 최종 root 이동 후 joint 재기록, 16 env | 첫 reset 16개 invalid, 추가 1개; 이후 전부 준비 |
| sleep 비활성화 / 최초 중복 root 기록 생략 / 링크 조회 선행, 각 8 env | 최초 8개 invalid가 그대로 재현됨 |
| 대기 위치에서 default reset 및 물리 예열, 8 env | 최초 8개 invalid; 이후 전부 준비 |
| 대기 위치 root 1 cm 이동 및 물리 예열, 8 env | 최초 8개 invalid; 이후 전부 준비 |

따라서 최초 박스 튐을 롤러 마찰만으로 설명할 수 없으며, 단순 예열이나 joint 재기록은
검증된 해결책이 아니다. 이 실험용 변경은 실제 학습 환경에 적용하지 않았다.
실제로 선반에서 수십 cm 벗어나는 현상이므로 shelf/footprint 검사 기준도 완화하지 않았다.
현재는 invalid reset으로 거부하고 재배치한 뒤 준비된 상태부터 SAC 데이터를 수집한다.

추가로 근본 수정이 필요하면 첫 teleport의 raw PhysX 링크 transform과
첫 physics substep별 움직임을 분리 계측하고, root 및 4개 flap의 이동이 일치하는
reset 방법을 검증해야 한다. 수정의 통과 기준은 최초 invalid 감소와 선반 위 안정화이며,
실패 검사를 끄는 방식으로 통과시키지 않는다.

## 변경 파일과 확인

- `src/kuavo_isaaclab_scene/rl/multi_box/training_env_cfg.py`: patch 버퍼 2배.
- `src/kuavo_isaaclab_scene/rl/runners/train_asymmetric_sac.py`: 초기 안정화 제한 시간 후
  준비 비율 확인 및 미준비 행 제외 안내; 롤러에만 원인을 돌리던 주석 수정.
- `tests/test_rl_asymmetric_sac.py`: 준비된 환경이 90%인 경우 진입하는 회귀 검사.
- CPU SAC 관련 검사 11개 통과. 실제 GPU3 첫 iteration의 보상 합 오차는
  약 `3.73e-9`, nonfinite transition은 0개였다.

GPU1 진단 프로세스는 모두 종료했다. GPU3 학습과 기존 Drive 관리자는 계속 실행하며
다른 사용자의 프로세스는 변경하지 않았다.

## 로그 위치

- 이전 시도들과 watchdog:
  `artifacts/rl/drive_runs/mbv2_grasp_sac_gpu3_1664_watchdog_20260922/`
- 현재 실행:
  `attempt_20260922_135230_19824/sac_20260922_135236_cefa58/`
  (위 watchdog 폴더 아래)
- 실패 시도의 정확한 종료 예외: 각 실행 폴더 `verification.json`의 `run_status.error`.
- 현재 진행: 현재 실행 폴더의 `console.log`, `metrics.jsonl`, `resources.jsonl`과
  부모 폴더 `status.json`. 이 문서의 수치는 작성 당시의 스냅샷이다.
- GPU1 재현 코드 및 원시 측정:
  `artifacts/rl/reset_diagnosis_20260922/probe.py`, `*.jsonl`, `*.console.log`.
