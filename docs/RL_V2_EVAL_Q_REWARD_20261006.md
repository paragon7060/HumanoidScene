# 대표 평가의 Q 표시 영상과 보상 설정

[Notion 중간 보고](https://app.notion.com/p/3f163918d42a817aa98cec7e2114034e)에
동일한 대표 평가 정책의 Q 표시 영상 4개를 첨부했다. 성공 2개와
랙 충돌·시간 초과 각 1개다. 전체 평가 결과 **30/128**은 그대로다.
대표 성능·학습 방법은 [중간 보고](RL_V2_GRASP_INTERIM_SUMMARY_20261006.md)를 참고한다.

처음 보는 사람은 Notion의 **중간 왼쪽 성공 → 상단 오른쪽 충돌** Q 영상을
차례로 보면 된다. 화면 위의 `r`은 지금 실행한 동작의 보상이고 `Qmin`은
앞으로 얻을 할인 누적 보상의 예측이다. 아래 그래프는 두 값의 시간 변화가
아니라 **예측 누적 보상과 실제 경로의 누적 보상**을 비교한다.
이 네 영상은 대표30/128 정책의 기록이며, 현재 비교 학습의 정책 영상이 아니다.

## 성공 사례의 시작 위치

영상은 base 위치·방향, 박스 위치·방향과 주변 박스 배치를 바꾼 평가에서 나온다.
구역별 기준 시작 자세에 더한 초기 배치 값은 다음과 같다.

| 성공 영상 | 좌우 이동 | 랙 바깥쪽 이동 | 회전 |
| --- | ---: | ---: | ---: |
| 중간 왼쪽 env 0 | +9.32cm | +11.32cm | −13.82° |
| 중간 오른쪽 env 5 | −19.29cm | +17.68cm | +4.78° |
| 상단 왼쪽 env 42 | +2.32cm | +4.72cm | +2.69° |

이는 샘플링한 rack 기준 오프셋으로, 최종 파지 위치 오차나 world 좌표가 아니다.
중간 선반은 좌우 ±20cm·바깥쪽 3–25cm·회전 ±15°,
상단 선반은 좌우 ±8cm·바깥쪽 3–10cm·회전 ±5° 범위다.
base가 초기 위치에서 작업 위치까지 제어기로 이동하고, 이후 위치를 유지하며
학습한 팔·상체·그리퍼로 잡는다. 서로 다른 초기 위치를 그대로 유지한 채
잡았다는 뜻이나, 임의의 먼 위치에서의 이동을 SAC가 학습했다는 뜻은 아니다.

## 영상의 Q와 실제 보상

대표 정책의 actor 업데이트 1204 / critic 업데이트 6864 체크포인트를 사용했다.
각 실제 관측에서 평가 당시 deterministic 목표를 복원한 뒤,
저장된 두 online critic에 정확히 같은 관측·목표를 입력했다.
목표를 실제 제어 명령으로 변환한 결과와 기록된 명령의 최대 차이는
**0.00002385**였다. checkpoint 모델 tensor도 모두 일치했다.
저장된 명령에서 포화된 목표를 역추정해 critic에 입력하지 않았다.

영상 위에는 `Q1`, `Q2`, `Qmin=min(Q1,Q2)`, 실제 즉시 보상 `r`을 표시한다.
그래프의 파란 선은 Qmin, 주황 선은 해당 평가 경로의
실제 할인 누적 보상 `G_t=Σ gamma^k r_(t+k)`이다. 할인율은 **0.999**,
learner reward scale은 **1.0**, entropy backup은 꺼져 있다.
접촉 보상 프로필에서는 시간 초과도 Q의 종료 상태로 다뤄 마지막 이후를 더하지 않는다.

Q는 미래 보상의 기대값이며 성공 확률이 아니다. 주황 선은 평가 종료 후 계산한
한 경로의 결과다. 기대값과 한 경로의 결과가 항상 일치해야 하는 것은 아니며,
이 4개 예시로 전체 critic의 보정 성능을 평가하지 않는다.
성공 예시에서도 초반 Q가 실제 미래 보상을 낮게 예측하므로 Q만으로 성공을 판단하지 않는다.

이 영상은 원래 학습 정책을 별도 물리 backend에서 평가한 대표 실행이다.
원본 critic은 GPU PhysX의 Q이고 영상·실제 return은 CPU PhysX에서 얻었다.
같은 checkpoint의 출력을 정확히 표시했지만 Q 학습 MDP와 평가 MDP가 같다는
뜻은 아니다. 따라서 이 차이를 SAC의 학습 부족이나 critic 학습 오차로 단정하지
않고, 이 평가 전이를 원래 Q replay에 넣지도 않는다. 현재 새 SAC 실행은
CPU PhysX용 Q를 새로 학습하며 자신의 같은 backend 평가로 비교한다.

영상 자세는 행동 실행 직후, Q는 해당 행동 실행 직전의 관측 기준이다.
base 접근 중에는 held-grasp critic의 학습 범위를 벗어나므로 `Q: N/A`를 표시한다.
원래 평가의 실제 자세 영상·행동·관측을 사용했고 물리 시뮬레이션을 재실행하지 않았다.
원본 HDF·영상·checkpoint SHA256이 변하지 않았고 optimizer와 replay는 수정하지 않았다.
모든 영상은 H.264/avc1·yuv420p·faststart로 저장하고 전체 디코딩을 확인했다.

| Q 표시 영상 | 첫 파지 구간 Qmin → 마지막 Qmin | 마지막 실제 보상 r | 결과 |
| --- | ---: | ---: | --- |
| 중간 왼쪽 env 0 | −0.222 → +5.964 | +6.996 | 성공 |
| 상단 왼쪽 env 42 | −0.218 → +6.077 | +6.996 | 성공 |
| 상단 오른쪽 env 3 | −0.152 → −1.323 | −5.973 | 랙 충돌 |
| 상단 왼쪽 env 2 | −0.274 → −0.832 | −0.038 | 시간 초과 |

수치는 영상의 control-step trace에서 가져왔다. 예를 들어 env0의 성공 순간에는
`Q1=+6.152`, `Q2=+5.964`, `r=+6.996`이다. 성공 이벤트 가중치+8과
실제 마지막 보상이 다른 이유는 접촉 잠재값의 종료 처리와 다른 보상·비용도
같은 동작에 합산하기 때문이다. 마지막 Q와 마지막 r이 같은 값일 필요도 없다.

## 실제 사용한 reward

대표 평가와 정상 종료한 GPU3 주 학습의 `physical_contract.reward_profile`은 동일하다.
현재 비교 학습에서는 포획 거리 스케일만10cm→2.5cm,
양손 합성만 평균→약한 손을 반영한 식으로 바꿨다. 가중치는 같다.
팔 탐색과 출력 포화 완화는 별도의 행동 샘플링·actor 손실 변경이며 보상 항목은 아니다.
[비교 설정](RL_V2_LEARNED_WORKPLACE_PRECISION_20261006.md)에 따로 기록했다.
파일의 일반 기본값보다 **이 실행의 agent.yaml → physical_contract.reward_profile**이
실제 사용값이다. 제어 반경·body controller도 agent.yaml의 전체 learner 계약과
checkpoint의 goal_contract를 기준으로 확인한다. manifest.json에 상속된 일반
goal_contract만으로 새 실제-flap controller의 반경을 추정하지 않는다.
접촉 프로필은 [contact_profile.py](../src/kuavo_isaaclab_scene/rl/multi_box/rewards/contact_profile.py)의
`contact_reward_weights()`가 기본값의 파지 이벤트·성공·파괴적 실패 가중치를 수정한다.
조합 식은 [model.py](../src/kuavo_isaaclab_scene/rl/multi_box/rewards/model.py),
실행 시 적용은 [v2_grasp.py](../src/kuavo_isaaclab_scene/rl/multi_box/managers/v2_grasp.py)에 있다.

| 항목 | 실제 가중치 | 계산 방식 |
| --- | ---: | --- |
| 양손 접근 | 2.0 | `gamma*approach_next−approach_prev` |
| 랙 앞 진입 | 1.0 | `gamma*front_next−front_prev` |
| 닫힘 축 정렬 | 0.5 | 근접 계수 × `(alignment_next−alignment_prev)` |
| flap 포획 위치 | 0.5 | `gamma*capture_next−capture_prev` |
| 물리 접촉 품질 | 1.0 | `gamma*quality_next−quality_prev` |
| 양손 파지 후 들기 | 0.4 | `gamma*lift_next−lift_prev`, 양손의 서로 다른 flap 파지 필요 |
| 한 손 / 양손 파지 이벤트 | 0.5 / 2.0 | 실제 파지가 성립하는 이벤트 |
| 최종 성공 이벤트 | 8.0 | 안전 위반 없이 실제 양손 파지·들기 조건 충족 |
| 닫힘 자체 | 0 | `jaw_gap_progress=0` |
| 미숙한 닫기 비용 | −0.002 | 준비 없이 닫는 정규화 크기에 곱함 |
| 진입 / 접근 거리 비용 | 각 −0.002 | 각각 `1−현재 점수`에 곱함 |
| base 이동 / 행동 변화 / 관절 한계 비용 | −0.0002 / −0.0001 / −0.002 | 각각 정규화한 크기에 곱함 |
| 로봇–랙 / 로봇–장애물 충돌 | −6 / −4 | 각각 10N / 5N을 넘으면 실패·감점 |
| 박스 낙하·과도한 들기·과속 / 작업영역 이탈 | −12 / −12 | 실패·감점 |

손별 접근 점수는 `s=exp(−거리/0.22m)`이며
`approach=0.25(sL+sR)+0.5*min(sL,sR)`다. 접근 거리는 flap 표면 거리이고,
관측의 상대 위치·방향은 flap 중간지점을 사용한다. 서로 다른 flap 조합 중
가까운 조합을 선택하며 배정 변경·reset에서 가짜 진행 보상이 생기지 않게 처리한다.
정렬은 flap 법선과 그리퍼 닫힘 축의 cosine 제곱을 근처에서 반영한다.
접촉 품질은 flap 영역 안의 서로 반대인 finger pad의 실제 힘을 5N에서 포화시키고,
약한 pad와 약한 손을 반영한다. 닫기 명령이나 거리만으로 접촉 보상을 받지 않는다.

성공·실패·시간 초과에서 접촉 잠재값은 0으로 처리한다.
그 외 접근·포획 등의 종료 잠재값은 이 대표 영상과 기존 비교 실행에 남아 있었다.
이를 Q의 종료 처리와 맞추고 정렬도 하나의 잠재값 차이로 계산하는
[선택형 새 보상](RL_V2_ABSORBING_GEOMETRY_20261007.md)을 별도의 GPU3 실행에 적용했다.
실제 reward manager 적용을 확인한 뒤 해당 비교의 전체 DEV128도 마쳤지만
자신의 초기23/128에서 학습 후1/128로 떨어졌다. 단순한 종료 보상 수정만으로
개선되지 않아 [실제 종료 TRAIN의 Q 학습을 보강한 비교](RL_V2_TERMINAL_CRITIC_CREDIT_20261007.md)를
GPU3에서 시작했다. 새 비교는 첫 초기 평가에 진입했고 아직 학습 후 개선을
확인하지 않았다. 이 종료 보상 비교와 원래 대표30/128 정책을 구분한다.
기존 영상의 `r`, Q와 성능 수치를 소급 변경하지 않는다.
한 동작의 `r`은 모든 항목의 합이므로 성공 이벤트 +8과 최종 `r`이 같지는 않다.
성공과 위험이 동시에 성립하면 성공 보상을 주지 않는다.
이번 실행의 자기 충돌 판정은 꺼져 있어 설정에 남은 자기 충돌 가중치 −4는 활성화되지 않는다.

## 재사용

종료된 평가의 원본 HDF와 영상, 평가 당시 checkpoint가 모두 필요하다.
exporter는 해당 run의 `launch.json`과 checkpoint 경로·업데이트 횟수,
영상과 episode의 배치, 프레임 수와 행동 대응을 확인한다. 다른 정책의 Q를 덧씌우지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/export_eval_q_videos.py \
  --run-dir /absolute/path/to/closed-evaluation-run \
  --checkpoint /absolute/path/to/evaluation-checkpoint.pt \
  --env-indices 0 42 3 2 \
  --output-dir /absolute/path/to/new-unique-media-directory
```

MP4와 preview PNG, control step별 Q·r·G trace JSON, 검증 JSON을 출력한다.
현재 실제-flap bounded held-base 정책(518/577/21차원)용이며 Isaac/GPU를 사용하지 않는다.
동일한 차원을 사용하는 [표준편차 축소 정책](RL_V2_GAUSSIAN_SERVO_20261007.md)도
저장된 고유 계약과 자기 Q 입력 변환을 복원해 표시한다. 그 형식을 지원하는 것은
영상이 이미 생성됐다는 뜻이 아니며, 위4개는 대표30/128 정책의 영상 그대로다.
[검증 근거](assets/rl_v2_eval_matching_Q_evidence_20261006.json)에 수치와 SHA256을 기록했다.
