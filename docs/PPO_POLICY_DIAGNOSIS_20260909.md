# PPO 정책 진단 — 2026-09-09

대상은 새 EEF/보상으로 학습한 `model_716.pt`다. 학습 중인 모델 파일을 수정하지
않고 고유한 진단 폴더로 복사한 뒤 고정해 사용했다. 환경과 보상을 바꾸는 처방은
이번 정책 분석의 범위에 넣지 않았다. 수치상 문제와 원인 가설을 구분한다.

## 재생 조건과 결과

- GPU 1만 `CUDA_VISIBLE_DEVICES=1`, 내부 cuda:0, Kit 물리 renderer 1로 사용.
- 기존 PPO 16384-env 학습은 계속 유지. 진단은 32 env, 그룹당 8개, seed 42,
  450 step = 15초. 정책/정규화는 eval mode, optimizer update 없음.
- mean: 평균 action; stochastic: 학습과 같은 독립 Gaussian sampling;
  quarter_noise: 같은 평균과 표준편차의 1/4; hold: action 0.
- 관측 227차원, 증분 action 16차원, 실제 충돌면 중점, 마찰 20/16,
  오른손 flap 파지 및 반대손 지지 허용, 장애물 0.1N 판정을 유지했다.
- 공유 폴더의 별도 작업이 02:29에 flap/보상 소스를 수정했음을 발견했다.
  보관된 source.diff를 적용해 원래 소스를 복원했고, 세 파일 모두 PPO 시작 시
  SHA256과 일치했다. 최종 `evaluation_original`에서만 이 사본을 import했다.
  공유 파일은 덮어쓰지 않았으며, 최종 설정의 closing.params는 원래의 빈 dict다.
- 앞선 `evaluation`은 카메라 점검, `evaluation_camera_fixed`는 바뀐 보상 소스를
  읽은 중간 실행이다. 두 프로세스는 종료했고 아래 수치에는 포함하지 않았다.

각 action 직전의 실제 PhysX 상태를 측정하고 안착 전 표본을 제외했다. 거리 평균은
마지막 5초, 최소 거리는 각 env의 최솟값을 다시 평균한 값이다. 준비/파지는 **step×env
표본 수**이며 에피소드 성공률이 아니다. 동일 초기 상태의 8개 복제이므로 독립적인
8개 상황에 대한 일반화 검증으로 해석하면 안 된다.

| 제어 | 마지막 5초 거리 | env별 최소거리 평균 | 파지 준비 표본 | 파지 표본 | 성공/완료 episode | clip된 action 성분 |
|---|---:|---:|---:|---:|---:|---:|
| mean | 4.13 cm | 3.19 cm | 0 | 0 | 0/8 | 64.8% |
| stochastic | 4.59 cm | 2.54 cm | 1 | 0 | 0/8 | 66.5% |
| quarter_noise | 4.13 cm | 3.26 cm | 0 | 0 | 0/8 | 65.3% |
| hold | 17.85 cm | 17.84 cm | 0 | 0 | 0/8 | 0.0% |

평균 정책에서 가장 가까운 상태는 3.18cm,
정렬도 0.9445(방향 오차 약 19.2도),
flap 법선 방향의 두 접촉면 간격 60.08mm였다.
그때 오른쪽 gripper signed target은 1.000
(+1 완전히 열림, -1 완전히 닫힘), 양면 파지 준비는 False였다.
실제 flap 두께는 약 3.17mm다. 간격은 flap 법선 투영값이며 그리퍼 고유 축의 개방 폭과
동일한 물리량은 아니다. 자세·목표 오차와 실제 접촉을 함께 봐야 한다.

![진단 그래프](../artifacts/rl/diagnostics/ppo_policy_20260909_022753_55cab0/policy_diagnosis.png)

## 정책 쪽에서 확인된 문제

### 1. 희귀한 준비 상태가 정책 정규화에서 82배 입력으로 바뀐다 — 직접 재현

관측 index 192는 `right_ready_to_close`다. checkpoint normalizer의 평균은
0.000004815, 표준편차는 0.002194이고 eps는 0.01이다. 따라서 0/1 비트가 1이 되는
순간 `(1-mean)/(std+eps)`가 **82.005**가 된다. 일반 재생의 정규화 입력 절댓값은
대부분 약 2~3인데, 유일한 준비 순간(step 127, env 14)에는 82.005로 뛰었다.
그 순간 오른쪽 gripper의 평균 raw action은 **+10.444**였고 +1로 clip되어 열기를
명령했다. 한 오른팔 성분은 -27.331이었다. 유효 접촉은 여전히 0이었다.

저장된 step 120의 같은 관측에서 이 비트 하나만 0→1로 바꾸는 CPU 민감도 시험도 했다.
최대 raw action 변화는 **26.65**, 그리퍼 평균 출력은 +1.888→+10.741이었다.
동일한 시험에서 이 feature의 정규화 후 값을 1로 제한하면 최대 변화는 0.685였다.
이 시험은 정책 입력 민감도의 인과적 비교이며, 실제 파지 조건을 만족시킨 물리 재생이나
수정 정책의 성공 검증은 아니다. 여전히 열기 출력이므로 clipping만으로 성공하진 않는다.

**우선 정책 전처리에서 boolean/flag를 running z-score 대상에서 분리하거나,
정규화된 입력의 범위를 제한하는 방안을 비교해야 한다.** 환경 관측 계약 자체를 바꾸지
않고 정책 내부에서 처리할 수 있다. 기존 가중치는 이 정규화에 적응해 있으므로 변경 후
평가·재학습이 필요하다. `rare_flag_sensitivity.json`에 수치가 있다.

### 2. 평균 action 자체가 범위 밖으로 치우친다 — 직접 확인

잡음을 완전히 제거해도 action 성분의 **64.8%**가 [-1,1] 밖이었다. 오른쪽 gripper는
평균 정책의 행동 가능한 표본 중 **99.4%**에서 +1보다 큰 열기 출력이었다.
그리퍼 target은 모든 유효 표본에서 +1로 유지됐다. 즉 탐색 잡음만 줄여서는 이미
열기로 치우친 평균 행동을 고칠 수 없다. 실제 접촉면은 대부분 flap 평면의 양쪽에
있지만 목표에서 수 cm 떨어지고 크게 열린 채여서 접촉이 없다.

이는 정책의 평균 출력까지 경계를 인식하도록 만드는 bounded distribution 또는
경계 인지 log-likelihood/regularization을 검토해야 한다는 증거다. 평균 출력 편향의
학습 원인 전체를 이 재생만으로 분리한 것은 아니다.

### 3. 탐색 분산 증가와 행동 포화 — 직접 확인

초기 표준편차 0.15에서 checkpoint 평균 1.1508, 관절별 1.015–1.284로
약 7.67배 커졌다. 오른쪽 gripper는 1.077이다. 실제 실행 범위는 [-1,1]이다.
확률적 재생에서 전체 action 성분의 66.5%,
오른팔 성분의 53.8%,
오른쪽 gripper의 71.6%가 범위를 넘어 clip됐다.
이는 0.02rad/step 팔 제어에서 정밀한 작은 보정 대신 경계 증분이 자주 실행된다는 뜻이다.

설치된 RSL-RL 3.1.2는 제한 없는 Gaussian의 raw action과 log probability를 저장하고,
Isaac wrapper는 실행할 때만 clip한다. entropy는 실행 후 분포가 아닌 Gaussian entropy다.
현재 entropy coefficient는 0.005이고 표준편차 상한/감쇠 schedule이 없다.
Gaussian entropy는 std를 키우면 계속 커지지만 실행 action은 이미 경계에 묶일 수 있다.
이 조합은 관찰된 분산 증가를 설명할 수 있는 정책 loss 측 원인 후보다.

다만 clip 자체를 PPO의 수학적 구현 오류라고 단정하지 않는다. latent action에 대한
정책경사는 성립할 수 있으며, 경계를 고려하지 않는 추정량의 불필요한 분산을 줄이는
것이 CAPG의 연구 주제다. 현재 entropy 항이 실패에 기여한 정도는 같은 환경에서
비교 학습해야 확정된다. [CAPG 논문](https://proceedings.mlr.press/v80/fujita18a.html)

**init_noise_std만 낮추고 resume하면 저장된 std가 복원되므로 해결되지 않는다.**
실제 std를 명시적으로 재설정하거나 학습 중 log-std 범위를 제한해야 하며, entropy
계수/스케줄도 정책 loss에서 조절해야 한다. tanh를 실행 action에만 덧붙이는 처방도
불충분하다. 선택한 분포에 맞는 log probability·entropy 처리가 함께 필요하다.

### 4. 상태와 무관한 독립 탐색 — 구조 확인, 인과 효과는 추가 검증

정책은 227→256→256→128→16 ELU MLP이고 `state_dependent_std=false`다.
각 관절의 std는 서로 다르지만 상태에 따라 바뀌지 않으며, 매 1/30초마다 독립 표본을
뽑는다. 접근할 때 크게 탐색하고 접촉 직전에는 정밀하게 탐색하는 분산 조절을 현재
구조가 직접 표현하지 못한다. 관절 평균들은 같은 MLP로 연동될 수 있으므로
“관절 협응을 전혀 표현하지 못한다”는 주장은 아니다.

후보는 양수 범위가 보장된 state-dependent log-std, 팔/그리퍼별 분산·entropy 제어,
또는 시간적으로 연속적인 gSDE 방식이다. gSDE는 로봇에서 매-step 독립 잡음의
거친 동작을 줄이기 위해 연구된 방법이지만 현재 task의 성공을 보장하지는 않는다.
단순 noise smoothing을 추가할 경우에도 PPO의 저장/재계산 확률 처리를 일치시켜야 한다.
[gSDE 논문](https://proceedings.mlr.press/v164/raffin22a.html)

### 5. 짧은 수집 구간과 매우 큰 minibatch — 설정 확인, 우선순위는 위 항목보다 낮음

16384 env × 32 step = rollout당 524288 transition을 수집하지만 5 epoch × 4 minibatch,
즉 20 optimizer step이다. minibatch 하나가 131072개다. 한 env의 수집 구간은 1.07초이고
PPO는 이 rollout을 업데이트한 뒤 버린다. 많은 총 샘플 수가 다양한 파지 경험이나 많은
정책 수정 횟수를 뜻하지 않는다. 다만 큰 batch가 그 자체로 잘못된 것은 아니다.

현재 gamma=0.99, lambda=0.95라서 30 step 뒤 TD 잔차의 직접 GAE 가중치는 약 0.159다.
critic bootstrap으로 그 뒤 미래도 반영하므로 “1초 이후를 전혀 배우지 못한다”는 뜻은
아니다. 64/128 step rollout이나 minibatch 16/32 분할은 비교 후보이며 메모리와 KL을
함께 측정해야 한다. 아직 이 설정을 실제로 변경해 학습한 결과는 없다.
[GAE 논문](https://arxiv.org/abs/1506.02438), [PPO 논문](https://arxiv.org/abs/1707.06347)

## 아직 원인으로 단정할 수 없는 항목

- 모든 checkpoint 모델 tensor는 유한했다. 수치 폭발/NaN을 실패 원인으로 볼 증거는 없다.
- MLP가 너무 작거나 LSTM/Transformer가 필수라는 증거는 없다. 관측에는 관절 속도,
  이전 action, 적분 목표값 등이 포함돼 있으므로 단순히 feed-forward라는 이유만으로
  이력이 전혀 없다고 말할 수 없다.
- 한 Gaussian이 여러 행동 모드를 표현하는 데 한계가 있어도, 이 작업의 한 가지
  성공 경로를 표현할 수 없는 것은 아니다. Diffusion 교체가 반드시 필요한 상황이라고
  판정하지 않는다.
- 학습률은 checkpoint 순간 1e-5였지만 이후 로그에서 다시 커졌다. 계속 하한에 묶였다고
  주장하지 않는다. 현재 logger에는 KL, PPO ratio clip fraction, critic explained variance가
  없어 업데이트가 지나치게 보수적인지/불안정한지는 충분히 판정할 수 없다.
- 잡음을 줄이는 inference 비교는 재학습 실험이 아니다. 닫기를 배우지 못한 평균 정책이
  낮은 잡음만으로 성공 정책이 되지는 않는다. 실패 전체를 policy만의 탓으로 분리한 것도 아니다.

## 다음 실험 우선순위 — 환경과 task 보상을 그대로 둘 경우

1. PPO를 유지하고 **희귀 flag 정규화 처리**, **평균/분산의 행동 경계 처리**,
   **분산 제한/재설정 + entropy 감소**를 우선 별도 실험한다. 모든 항목을
   한꺼번에 바꾸지 않고 같은 평가 조건에서 준비 빈도, 양면 접촉, 파지 지속 시간,
   성공률, action 포화율, KL을 비교한다. 기존 학습에 해당 변경을 적용하지 않았다.
2. 다음으로 상태별/그리퍼별 탐색과 minibatch·rollout을 각각 비교한다. 네트워크 크기
   확대는 파지 동작을 표현하지 못한다는 추가 증거가 생겼을 때 우선한다.
3. VR에서 가능한 성공 동작을 **시범 데이터로 기록해 BC로 정책을 초기화한 뒤 PPO로
   미세조정**하는 경로가 가치 있다. 사용자 확인은 물리적 가능성의 증거지만 자동으로
   학습용 시범 데이터가 만들어졌다는 뜻은 아니다.
4. 동작 시퀀스가 필요하면 VR 시범→Diffusion Policy BC→DPPO를 비교한다. Diffusion은
   복수 행동 모드와 연속 action sequence를 다루는 장점이 있고, DPPO의 강한 결과는
   주로 시범으로 사전학습한 정책의 미세조정에서 나왔다. 랜덤 초기 diffusion이 자동으로
   PPO보다 쉽게 파지를 찾는다고 기대하지 않는다.
   [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/),
   [DPPO](https://arxiv.org/html/2409.00588v3),
   [시범 활용 정책경사 연구](https://arxiv.org/abs/1709.10087)

기존 native diffusion BC/DPPO는 227차원 관측과 동일한 정규화 증분 action 계약의
성공 episode 데이터가 필요하다. 임의 LeRobot checkpoint를 그대로 load하는 구조는
아니다. LeRobot을 데이터/학습 관리에 쓰려면 관측·행동 표현을 맞추는 연결 작업이 필요하다.

## 재현 자료

- 실행 코드: `scripts/rl/diagnose_ppo.py`
- checkpoint SHA256: `22565d956266ed1e424adb3edd3a09eef0cd45b47da2f8e3f06310b1bc3609a4`
- 진단 폴더: `artifacts/rl/diagnostics/ppo_policy_20260909_022753_55cab0`
- 최종 재생: `evaluation_original/metrics.json`, `rollout.npz`, `observations.npz`, `console.log`.
- 영상: `evaluation_original/policy.mp4`. 실제 PhysX pose와 USD mesh의 CPU 미리보기이며,
  RTX 화면 녹화가 아니다. 성공 예시 영상도 아니다.
- 분석: `analysis.json`, `policy_audit.json`, `rare_flag_sensitivity.json`, `policy_diagnosis.png`.
- 시작 시 소스: `original_source/`, `historical_source_audit.json`.
