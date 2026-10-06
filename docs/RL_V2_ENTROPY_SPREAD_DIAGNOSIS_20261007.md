# SAC 출력 포화와 목표 엔트로피

출력 포화 완화 비교도 **29→24→12/128**로 개선되지 않았다.
두 번째 전체 평가의 구역별 성공은10/2/0/0이다. 실제 양손 파지·유지·들기12건,
랙 충돌62·박스 낙하6·과도한 들기6·시간 초과41·초기화 무효1건을 확인했다.
[전체 평가](assets/rl_v2_mean3_second_full_DEV_after768_20261007.json).

실제 TRAIN actor1898/Q9640의 마지막 배치에서는 몸체 평균 최대 절댓값8.418,
활성4,864좌표 중462좌표가3을 넘었고 연속 탐색 계수alpha는5.65e−6이었다.
Mean3 손실이 적용됐지만 포화를 모두 막지는 못했다. 이 배치의 수치를
전체 구역·경로의 포화율로 해석하지 않는다.

## 현재 계산에서 확인한 점

실제 목표는 `anchor + available_radius × tanh(mean + std × noise)`다.
Gaussian std0.03–0.12를 유지해도 mean이 커지면 실제 목표의 변화가 줄어든다.
현재 목표 엔트로피에도 **mean 위치의 tanh Jacobian**을 더한다. 작은 std 상한과
포화된 초기 시연 정책에서 달성할 수 없는 엔트로피를 계속 요구하지 않게 하기
위한 설계다. 부호 오류나 affine Jacobian 누락을 발견한 것은 아니다.

다만 mean이 범위 끝으로 몰리면 목표 엔트로피도 낮아진다.
`logp + target_entropy`가 양수면alpha를 올리고 음수면 내리는 코드에서,
큰 mean의 sample·target Jacobian 변화가 서로 상쇄되므로 latent std가
충분하다는 이유로alpha를 내릴 수 있다. 실제 목표의 탐색 폭을 회복하는 요구는 약하다.
현재 초기alpha1e−5, 최소1e−7, 최대0.001, entropy optimizer lr0.0003이다.
Q의 entropy backup은 꺼져 있다.

## 현재 코드로 수행한 분포 진단

Isaac·GPU·실제 물리 동작 없이 현재 `BoundedCorrectionHybridSAC`를 사용했다.
활성 좌표를 같은 조건으로 분리하고 std0.112·가용 반경0.30에서 mean만 바꾸어
조건마다16,384개 Gaussian을 샘플링했다. Float64로 FP32의 추가 포화 오차를
피했다. 표의 단위는 **정규화된 관절 목표**이며 rad·m 또는 실제 이동거리가 아니다.

| pre-tanh mean | 샘플 목표 표준편차 | 현재 alpha 갱신 신호/좌표 |
| ---: | ---: | ---: |
| 0 | 0.0331513 | −0.4318 |
| 1.5 | 0.00624172 | −0.4421 |
| 3 | 0.000343857 | −0.4443 |
| 6 | 0.000000857006 | −0.4444 |
| 8 | 0.0000000156968 | −0.4444 |

Mean0→3에서 목표의 표준편차가약96분의1로 줄지만alpha 신호는 계속 음수다.
**현재 엔트로피 설정만으로 포화에 따른 탐색 축소를 막지 못할 수 있다.**
이 통제된 분포 진단은 새 성공률이나 현재 모든 관절의 실제 움직임이 아니다.
실제 mean3 비교의 반경은0.15로, 이 표의0.30 조건과 구분한다.
[수치·소스 SHA256·실제 배치](assets/rl_v2_mean_adaptive_entropy_spread_diagnosis_20261007.json).

## 다음 수정의 판단 기준

반경0.30 비교는 전체 초기23/128 이후 Q2,094회·held96,346행의 warmup을
통과해 actor12회 갱신을 시작했다. 해당 배치의 평균 최대0.051·포화0개다.
첫 학습 후 전체 DEV128과 실제 출력 통계로 범위 확대의 효과와 재포화를 구분한다.
[새 제어 비교](RL_V2_REANCHORED_BODY_CONTROLLER_20261007.md).

추가 후보는 **큰 mean에서 목표 엔트로피가 계속 낮아지지 않도록 바닥을 두는
선택형 설정**이다. Mean1.5의 Jacobian까지만 반영한 가상 계산에서는mean2
이상에서alpha 신호가 양수로 바뀌었다. Mean hard clip이나 관절 범위 축소가
아니다. 기존 std 상한·affine 경계·고정 좌표의 entropy 제외는 보존해야 한다.
현재 실행에는 적용하지 않았고 실제 파지 개선은 입증하지 않았다. 구현 시
목표 식·alpha 초기값과 checkpoint/replay 재개 계약을 명시해 별도 비교해야 한다.

왼손 접근·연속 진입·랙 충돌도 남아 있다. 이 진단은 실패의 유일한 원인이나
해결 보장이 아니다. 박스·base·배경·동적 flap 무작위화, 보상·성공·안전 기준과
독립 FINAL 분리는 유지한다.

소스: [목표 상수](../src/kuavo_isaaclab_scene/rl/algorithms/asymmetric_sac.py),
[mean에 따른 목표·alpha 갱신](../src/kuavo_isaaclab_scene/rl/algorithms/hybrid_goal_sac.py),
[경계와 affine Jacobian](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/actual_flap_residual_sac.py).
