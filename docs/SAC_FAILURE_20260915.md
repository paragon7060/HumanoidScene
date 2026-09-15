# 2026-09-15 전체 관절 SAC 실패 분석과 box 안전 종료

대상: `sac_20260914_210825_1f4e24`, GPU 3 / 4096 env / 25 actions.
453 iteration까지 기록됐고 다음 iteration의 Q 업데이트에서
`Non-finite training loss`로 중단됐다. 최종 로그와 checkpoint의 Drive
업로드·크기·MD5 검증은 완료됐다.

## 확인한 증거

| 증거 | 관찰 |
| --- | --- |
| 최초 평균 lift 1 m 초과 | iteration 250: 1.066 m |
| 거대한 stability 비용 | iteration 382: 평균 −6.3747×10¹⁵ |
| 박스 위치 폭주 | iteration 386: 평균 lift 9.6099×10⁸ m, 오른손 target distance 1.7386×10⁹ m |
| checkpoint 400/450 tensor | model·normalizer·optimizer 모두 유한 |
| checkpoint 450 normalizer | 평균 최대 3.3031×10⁷, 분산 최대 1.5977×10²⁰ |
| 마지막 기록된 Q loss | iteration 453: 0.04674 |
| 중단 위치 | twin Q MSE loss 계산 뒤, backward 이전의 유한성 검사 |

평균 lift는 모든 action-enabled step의 실제 PhysX 박스 중심 높이에서
환경 원점 및 각 박스의 reset 기준 높이를 뺀 값이다. reward용 0~1 clip
값이 아니다. 소수 환경에 거대한 좌표가 생겨도 전체 평균이 매우 커진다.
이는 파지·들기에 성공했다는 의미가 아니라 물리 상태가 비정상적이라는 증거다.

## 여러 policy에 공통인 환경·보상 문제

기존 실패 조건은 박스 바닥 낙하, 로봇 workspace 이탈, cargo loss 및
활성화한 경우의 충돌이었다. 박스의 상한 높이, 과속, 비유한 상태는
검사하지 않았다. 위로 튀거나 멀리 날아간 박스가 timeout까지 살아남을 수 있었다.

stability 보상은 박스 각속도의 제곱에 비례하며 상한이 없었다. 물리 상태
폭주가 극단적인 보상으로 변환되어 PPO/SAC/DPPO 모두 오염될 수 있는 구조였다.

최초 물리 폭주를 일으킨 접촉은 저장된 로그만으로 확정할 수 없다.
PlanarDrive의 root pose 직접 적분, 충돌에 의해 멈추지 않는 base 명령,
torso/arm PD target 변화, 고마찰 손가락(20/16), 잠긴 flap의 접촉 조합은
추가 비교 실험 대상이다. 해당 조합을 원인으로 확정하거나 마찰을 낮추지는 않았다.

## SAC의 데이터·수치 문제

원래 replay 입력에는 유한성 검사가 없었고, normalizer는 action-enabled
관측을 그대로 통계에 포함했다. 박스 위치·속도·거리의 거대한 분산이
checkpoint에 남았다. 이후 정상 관측도 이 오염된 통계로 정규화된다.

유한한 입력과 기존 450 model을 사용해도, 단일 reward를 1e21로 넣으면
float32 MSE 제곱이 overflow하여 동일 오류가 재현됐다. 일반 크기의 reward는
정상 업데이트됐다. gradient clipping은 backward 이전 loss overflow를 막지 못한다.
큰 replay에는 과거 오염 transition이 남으므로 현재 iteration 지표가
정상적으로 보여도 뒤늦게 나쁜 sample을 뽑을 수 있다.

또한 terminal bootstrap의 `0 * NaN`은 NaN이다. 비유한 terminal next
관측/가치가 있으면 종료 sample도 오염될 수 있었다.

실제로 실패한 minibatch와 마지막 model/replay는 저장되지 않았다.
따라서 마지막 오류가 reward overflow인지, NaN 관측/normalizer인지
정확히 하나로 확정할 수 없다. 환경 폭주·보상 outlier·정규화 오염은 확인됐다.

## 적용한 변경

- 모든 box 중 하나라도 reset 기준 lift가 **0.50 m 이상**이면 failure 종료.
- 위치/quaternion/root velocity가 비유한 값이면 종료.
- box 선속도 **10 m/s**, 각속도 **100 rad/s** 초과 시 종료.
- 충돌 비용/실패 비활성 여부와 무관하게 적용. success보다 failure가 우선.
- settling 완료 시 비정상 높이를 새 baseline으로 capture하지 않음.
- 해당 실패 step의 shaping은 0; 기존 failure 비용은 유지.
- SAC에서 비유한 관측은 normalizer에서 제외하고 비유한 transition은 replay에서 제외.
  `nonfinite_transitions` 및 settling 제외 수를 구분해 기록.
- SAC soft target와 공통 GAE에서 terminal bootstrap을 `torch.where`로 차단.

공통 helper: `rl/mdp/box_safety.py`. 설정은 `rl/tasks/specs.py`의
`max_box_lift_height`, `max_box_linear_speed`, `max_box_angular_speed`에서 관리한다.
실험별로 `configure_task()`에서 바꿀 수 있다. 4박스 환경도 같은 helper를
사용하며 `rl/multi_box/spec.py`에 같은 설정을 제공한다.
Curriculum과 마찰은 변경하지 않았다. 물리 폭주 자체를 해결했다고 판단하지 않는다.

## 확인

관련 CPU 테스트 79개 통과. GPU 3의 4-env 실제 PhysX 평가에서 박스 하나씩
75 cm / 1000 m 높이 / 100 m/s 속도로 변경한 결과 종료 flags는
`[false, true, true, true]`, 보상은 정상 환경의 작은 비용 및
실패 환경 각각 −60이었다. reset 뒤 관측은 모두 유한했다.
검증 프로세스는 종료됐다.

수치 재현: `artifacts/rl/diagnostics/mobile_sac_numeric_reproduction.json`.
PhysX 검증 결과:
`artifacts/rl/evaluation/sac_all_joints_450_20260915_141033/guard_probe/play_20260915_141239_7c5344/box_guard_probe.json`.

450 checkpoint의 영상은 당시 source snapshot과 task 설정을 그대로 재생한다.
새 안전 종료를 적용한 정책 재학습/성능 평가와 구분한다.
정규화 통계가 오염된 400/450 checkpoint에서 단순 재개하는 것은 피하고,
다음 학습은 새 안전 조건으로 fresh training하는 것이 적절하다.
