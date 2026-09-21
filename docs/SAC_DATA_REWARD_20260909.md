# SAC 데이터 수집·닫기 보상 수정 (2026-09-09)

## 이번 변경

- SAC는 **step 시작 전** settling readiness를 복사한다. 행동 불가능한
  transition은 replay와 관측 정규화에서 제외한다. 행동 가능한 terminal
  transition은 pre-reset next observation과 함께 보존한다. Settling 중의
  충돌 종료는 환경에서 계속 판정하고 episode 통계에도 남긴다.
- Warmup은 유효 transition 수와 `--warmup-vector-steps × num-envs` 중 큰 값을
  기준으로 한다. 일부 환경만 행동 가능한 step의 업데이트 수는 유효 비율에
  비례하며 settling/warmup 동안 미뤄 둔 업데이트를 나중에 몰아서 하지 않는다.
- 관측 정규화는 유효 데이터만 사용한다. Alpha, Q loss, reward 가중치와
  정상 제어 중 정규화의 온라인 갱신 방식은 이번 변경 범위에 포함하지 않는다.
- 파지/접촉 지표는 매 iteration 마지막 step 대신 **모든 유효 transition**의
  평균이다. 종료 직전 지표도 보존한다. `reward/*`는 가중치와 dt를 포함한
  항목별 실제 보상 기여이며, 유효 데이터가 없는 iteration은 null이다.

## Curriculum 없는 연속 닫기 비용

거리 d, 정렬도 a, flap 목표가 두 pad 사이 구간을 벗어난 거리 s로 준비도를 계산한다.

```text
p = exp(-(d / 0.10)^2 - (1-a) / 0.10 - s / 0.02)
desired_gap = flap_thickness + (1-p) × 0.04
score = p × exp(-abs(jaw_gap-desired_gap) / 0.02)
premature = (1-p) × clamp((flap_thickness+0.04-jaw_gap) / 0.04, 0, 1)
closing = -(1-score+premature)
```

선택된 파지 손에만 적용하고 settling 중에는 0이다. 기존 weight 0.25와
dt=1/30을 적용한다. 원래의 hard ready-to-close 관측과 실제 접촉·양면 파지
판정은 그대로다. 대기/빈 공간 닫기에 양의 보상은 없으며, 멀거나 정렬되지
않은 상태에서 닫으면 추가 비용이 생긴다. 목표에 접근·정렬하고 flap을 감싸면
목표 간격이 열린 간격에서 flap 두께로 연속적으로 줄어든다.

접근·정렬 비용, 실제 파지 후 lift/hold, prelift 흔들림 비용, 우측 한 손 flap
파지, 반대손 받침 허용, robot–rack/주변 장애물 **0.1N 초과 실패**는 유지한다.
Reset 자세·분포와 curriculum은 변경하지 않는다. 초기 박스–랙 안착 접촉은
로봇–장애물 접촉과 다른 판정이다.

## 새 본 학습

| 설정 | 값 |
|---|---:|
| 물리 GPU / 내부 device | CUDA_VISIBLE_DEVICES=2 / cuda:0 |
| 환경 | 4096 |
| Replay | 22,000,000 transition, CUDA |
| Replay 메모리 | 38.62 GiB (227-D obs, 16-D action) |
| 가득 찬 replay의 환경당 유효 시간 환산 | 약 179초 |
| 전체 VRAM 예상 / 자체 사용 상한 | 약 50 GiB / 52 GiB |
| 전체 GPU 여유 하한 | 8 GiB |
| Warmup | 유효 2,000,000개 및 유효 vector step 환산 450회 이상 |
| Batch / updates | 1024 / 전체 env가 유효한 step당 4회 |
| Rollout / iterations | 16 / 1500 (최대 총 98,304,000 raw transition) |
| Checkpoint | 50 iteration마다 및 정상 종료 시, replay 제외 |
| Drive | 설정된 Drive remote, 300초 간격 |

실제 사용량/성능은 실행 지표로 판단한다. 새 보상으로 fresh initialization을
사용하며 기존 checkpoint 300의 Q/optimizer를 복원하지 않는다.

실행 소스는 `artifacts/rl/source_snapshots/sac_datareward_20260909_023349_85abd312`에
고정했다. Python/config/launcher를 복사하고 기존 assets는 심볼릭 링크로 재사용한다.
Drive 인증은 복사하지 않고 원래 checkout의 업로더를 사용한다.

관리 폴더는 `artifacts/rl/drive_runs/sac_gpu2_datareward_20260909_023349_85abd312`다.
현재 진행률은 그 아래 `status.json` 및 `sac_*/metrics.jsonl`을 확인한다.
설정의 존재만으로 실행 중/완료라고 판단하지 않는다. 검증된 오래된 checkpoint만
정리하고 최신 2개 및 최신 검증본 2개를 보호한다. 종료 후 닫힌 로그도 업로드·검증한다.

검증은 요청에 따라 핵심 CPU 테스트 7개와 문법·diff 확인으로 제한했다.
별도 Isaac 제어/성능 검증 실행 없이 본 학습을 시작했다. 이 테스트들은
settling 샘플 필터, warmup, terminal 데이터 보존 및 보상 조건의 회귀 확인이며,
실제 파지 성공이나 새 보상의 성능 향상을 보장하는 결과는 아니다.
