# 실제 성공 경로의 SAC 가치 추정 확인

학습된 정책의 일부 성공은 확인했지만 원래 여섯 배치의 일반화는 아직
부족하다. 안쪽 파지점 v6의 TRAIN384 뒤 전체 평가128회는 성공11·안전
위반54·시간 초과63회였고 중형 성공은0회였다. 전체 점수를 바꾸거나 평가
자료를 학습에 넣지 않고, 종료된 기록과 해당 평가의 실제 모델을 대조했다.

## 실제 누적 보상과 Q

Actor695/Q갱신4,828의 같은 모델로 기록된 관측에서 실행 명령을 재현했다.
124경로는 기존 오차 기준을 통과했다. 중간 오른쪽small의 시간 초과4경로
(Env1·25·41·105)는 최대 오차0.00027–0.00028로 통과하지 못해 Q 분석에서
제외했다. 오차 기준은 완화하지 않았고 전체 평가 분모128도 유지했다.

| 실제 종료 | Q 분석 경로 | held 시작 Q 중앙값 | 실제 할인 return 중앙값 |
|---|---:|---:|---:|
| 양손 파지·유지·들기 성공 | 11 | 0.523 | 4.614 |
| 안전 위반 | 54 | -2.819 | -4.789 |
| 시간 초과 | 59 | -0.908 | -1.116 |

이 평가에서는 실패한 채 버티는 경로의 실제 보상이 성공보다 높지 않았다.
반면 Q는 성공 경로의 실제 return보다 낮았다. Q는 누적 보상의 추정값이며
성공 확률이 아니다. 실제 한 번의 return과 추정값 차이만으로 Bellman 계산
오류나 학습 실패 원인을 확정하지 않는다.

![실제 전체 평가의 Q와 return, 상단 왼쪽 성공 두 경로](assets/rl_v2_interior_contact_repaired_closed_DEV4_critic_diagnostic_20261009.png)

할인은0.999, reward scale은1, entropy backup은꺼짐이다. 그림의 파란선은
당시 모델의 min(Q1,Q2), 주황선은 실제 종료까지의 보상을 사후 계산한 값이다.
새 물리 평가·애니메이션·영상은 아니며 원래 성공률도 그대로다.

## 성공 경험은 critic에 연결되어 있음

같은 종료 체크포인트의 **실제 TRAIN 성공21경로/10,549행**도 검사했다.
완료된 성공만 들어가는 별도 bank가 있고, 당시 critic batch256행 중46행,
약17.9%를 이 bank에서 가져오는 코드가 연결되어 있었다. 따라서 성공 경험이
critic에서 빠졌다고 판단하지 않는다. 실제 TRAIN의 전체 할인 return을 쓰는
추가 critic 목적함수도 batch64·가중치0.1로 연결되어 있었다.

TRAIN 성공 경로 자체에서도 시작 Q 중앙값0.790과 실제 return4.791의 차이가
있었다. 기록된 경로의 return은 과거 행동의 결과이며 현재 정책의 정확한
기대 return과 같다고 가정할 수 없다. 더 긴 학습에서 이 차이와 실제 전체
성공률이 함께 줄어드는지 확인하고, 다음 비교의 후보는 실제 TRAIN return
목적함수의 가중치와 구역별 성공/실패 비중이다. 현재 실행의 가중치는 바꾸지 않았다.

같은 TRAIN 성공 경로의 종료 전16상태씩, 연속 action32표본을 추가 질의했다.
현재 greedy Q와 연속 확률 action·네 그리퍼 branch 기대 Q 차이의 경로별
중앙값은약-0.0054였다. 이 자료에서는 action 무작위성만으로 약4의 return
차이가 설명되지 않았다. 실제 물리 rollout이나 target network 변경은 없었다.

[실제 TRAIN 성공 bank·critic 연결·수치](assets/rl_v2_interior_contact_repaired_closed_TRAIN_success_critic_diagnostic_20261009.json) ·
[같은 상태의 정책 기대값 진단](assets/rl_v2_interior_contact_repaired_closed_TRAIN_policy_expectation_diagnostic_20261009.json).

## 초기 무효 조건과 상단 왼쪽 성공의 해석

v5·v6·v7의 첫 배치 guard 파일은 SHA256까지 같았다. 원래 요청128개 중
유효115·무효13개가 공통이다. 무효8개는 원래 background 박스가 안정적으로
선반 위에 있지만 지정 영역 밖에 있었고,5개는 초기 정착 중 실패하여
배치가 교체되었다. 교체 후 주차된 박스 자세는 원래 실패의 움직임이 아니다.
실패 이전 trace는 수집되지 않아 그5개가 왜 처음 실패했는지는 확정할 수 없다.

학습 후 상단 왼쪽small의 실제 성공은 Env78·90의2회다. Env78은 초기 무효,
Env90은 초기 유효 시간 초과였다. 두 사례 모두 실제 양손 opposing 파지·
0.267초 유지·proof lift·안전을 통과했지만, 초기의 유효 실패2개를 같은 물리
상태에서 개선한 것으로 해석하지 않는다. 공통 유효115조건 전체는14→10회였다.
무효 요청도 원래 분모에 남기며 교체 장면의 전이는 학습에 넣지 않는다.

[세 실행의 초기 guard와 실제 측정 근거](assets/rl_v2_predictive_initial_guard_comparison_20261009.json).

## 재현

종료 실행의 `status.json`·`verification.json`·관리자의 최종 Drive 검증,
원래 writer/supervisor 종료, 같은 평가의 보존 모델·checksum·계약·전체128조건을
모두 확인해야 한다. 현재 쓰는 HDF/replay나 다른 사용자의 실행은 읽지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/rl/audit_closed_dev_critics.py \
  --run-dir /absolute/path/to/closed-run \
  --checkpoint /absolute/path/to/matching-protected-checkpoint.pt \
  --matching-model-proof /absolute/path/to/matching-model-proof.json \
  --whole-eval-proof /absolute/path/to/completed-whole-DEV-proof.json \
  --wave 4 --output-dir /absolute/path/to/unique-analysis-directory \
  --plot-envs 78 90
```

출력은 Q·return·실제 종료의 진단이다. 원시 관측/action payload를 출력하지
않고 물리 재실행·optimizer 갱신·replay 입력·새 인증을 사용하지 않는다.
[전체 평가의 같은 모델·Q 분석 범위](assets/rl_v2_interior_contact_repaired_closed_DEV4_critic_diagnostic_20261009.json).
