# 실제 한 스텝 제어 명령을 평가하는 SAC Q 옵션

기존 비교는 초기 성능을 넘지 못했다. 팔 탐색20%는29→21→18→22→18/128,
출력 포화 완화는29→24→12→18→7/128이었다. 마지막7건은 중간 좌3·우4이며
상단 양쪽은0건이다. 랙 충돌73·박스 낙하6·시간 초과41·초기화 무효1건을
원래128개 분모에 포함했다. 실제 양손 접촉·유지·들기 조건으로 확인한 반복 DEV이며
독립 FINAL은 사용하지 않았다. Writer는 exit0으로 끝났고 원래 관리자가
10/07 05:36 KST에 닫힌 로그를 포함한 최종 Drive 체크섬 검증을 완료했다.
[마지막 평가](assets/rl_v2_mean3_final_full_DEV_after1536_20261007.json),
[종료·백업 근거](assets/rl_v2_mean3_final_backup_verified_20261007.json).

## 새로 확인한 문제

Actor306·Q3,272를 닫힌 파일로 보존하고 실제 성공 TRAIN15개/6,899행을 확인했다.
정규화된 목표 오차가 크더라도 실제 단위와 제어 명령을 따로 봐야 한다.
상단 오른쪽 성공 경로의 마지막64행에서 앞뒤 torso 목표 RMSE는0.57cm였고
일부 왼팔 관절 목표 RMSE는 약4.9–5.1°였다. 이것은 과거의 같은 성공 상태에서
계산한 목표 차이이며 실제 로봇의 흔들림·EEF 오차·새 성공을 측정한 것이 아니다.
[관절·상체별 목표 오차](assets/rl_v2_body_goal_groups_20261007.json).

기존 제어기는 관측한 pending drive 목표를 빼고, 관절별 한 스텝 변경량을
제한한다. 따라서 서로 다른 절대 목표가 동일한 즉시 모터 명령으로 변환될 수 있다.
그런데 기존 Q는 clipping 이전의 절대 목표를 그대로 입력받았다.

| 상단 오른쪽 마지막64행 | 왼팔 | 오른팔 | torso 앞뒤·높이 |
| --- | ---: | ---: | ---: |
| 한 스텝 제어 제한을 넘은 목표 좌표 비율 | 49.3% | 49.6% | 54.7% |
| 그 구간에 걸린 Q 목표 경사의 절댓값 비율 | 46.4% | 64.1% | 89.7% |

실제 production decoder와 telemetry를 사용했고, 모든6,899행에서 명시한
unclipped 식에 clamp를 적용한 결과가 기존 decoder와 정확히 같았다.
다른 목표와 성공 정답의 **전체 동작이 같다는 뜻은 아니다**. 각 좌표에서 이미
제한을 넘은 동안 Q가 물리 변화 없는 방향도 선호할 수 있다는 진단이다.
Deterministic body mean의 부분 경사이며 live stochastic 혼합 배치·공유 신경망·
entropy·그리퍼 손실의 전체 optimizer 방향을 측정한 것은 아니다.
이 분석만으로 학습 실패의 유일한 원인이나 새 옵션의 성공을 주장하지 않는다.
[실제 명령 변환·clipping·Q 경사 근거](assets/rl_v2_goal_servo_alias_20261007.json).

## 변경한 방법

새 checkpoint 형식 `staged_actual_flap_reanchored_servo_critic_hybrid_sac_v1`을
명시적으로 선택하면, Q가 **실제로 실행되는 body delta와 두 binary jaw**를 본다.
Actor·온라인 replay의 action은 원래 absolute goal21개로 유지하고 Q 입력 직전에
현재 actor 관측의 proprioception·pending target을 사용해 기존
`physical_body_actions.goal_body_command`로 변환한다.

```mermaid
flowchart LR
    A[관측에서 SAC 목표 생성] --> B[기존 제어기: pending 목표 차감·한 스텝 제한]
    B --> C[실제 모터 명령]
    B --> D[새 Q의 행동 입력]
```

One-step Q, 실제 n-step16 보조 Q, actor의 네 jaw 조합, 모든 target Q가 같은
변환을 사용한다. 종료 placeholder는 변환에 넣지 않는다. 제한 밖 좌표의
actor-Q 경사는 정확히0이 되고, 같은 명령을 만드는 목표의 Q 입력은 같아진다.
Actor의 affine goal 탐색·goal-space entropy·실제 controller는 유지한다.
이는 속도 제한이나 새로운 IK 동작을 바꾸는 수정이 아니다.

새 Q 입력은 기존에 학습한 Q·optimizer와 호환되지 않으므로 **학습 전 actor0·Q0,
빈 네 optimizer·critic 정규화·온라인 replay**에서만 준비한다. 실제 TRAIN 성공
목표·보상·결과·episode 은행은 그대로 사용하며 다시 붙이거나 변환해 저장하지 않는다.
학습한 기존 Q를 잘못 재개하면 계약 검사에서 거부한다.

박스·base 시작점·배경·단단하지만 움직이는 flap의 무작위화, 충돌10N/5N,
양손 실제 파지·유지0.25초·지지면 clearance8mm의 성공 조건을 유지한다.
보상·가중치는 [Q 영상과 reward 설명](RL_V2_EVAL_Q_REWARD_20261006.md)과 같다.
Curriculum·고정 박스·안전 기준 완화는 적용하지 않는다.

## 검증과 다음 비교

관련 고유 테스트89개가 통과했다. 같은 clipped 명령의 Q 입력 일치, 실제 clipping
경사의0, pending target 사용, 종료 placeholder 제외, 모든 Q 경로 적용,
기존 default 보존, 저장·복원과 잘못된 구형 Q 거부를 확인했다.

실제 TRAIN 관측698개에서 초기 몸체·jaw 출력이 bit 단위로 동일했고, 성공6,899행의
명령 변환은 유한·범위 내·같은 binary jaw였다. 실제 전체 training pilot에서도
엄격한 계약과 모델 tensor, 성공·n-step 은행6,899행을 복원했다. 업데이트·온라인
replay·네 optimizer state·critic 정규화 count는0이다.
[초기화](assets/rl_v2_servo_critic_initialization_20261007.json),
[실제 trainer 복원](assets/rl_v2_servo_critic_full_restore_20261007.json).

이는 연결·수학적 불변성 검증이며 새 파지 성능은 아직 검증하지 않았다.
기존 tail 비교의 첫 학습 후 DEV도 유지한다. 새 옵션은 자신의 전체 초기 DEV128을
기준으로 네 구역32개씩 평가하고 초기화 실패도 분모에 남긴다. 다른 실행의
flap 추첨·solver 이력까지 같다고 가정하지 않는다. 독립 FINAL은 최종 검증에 남긴다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_servo_critic_actor.py \
  --initial-checkpoint /absolute/path/to/closed-fresh-initial/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/matching/training_manifest.json \
  --waypoints /absolute/path/to/matching/waypoints.json \
  --output-dir /absolute/path/to/new-unique-servo-critic-inputs
```

학습은 기존 `batched_staged_goal_with_drive.py`와 GPU3 마스크를 재사용한다.
고유 실행 폴더·기존 Drive 연결·300초마다 checksum 검증·최신2개 보존·종료 후
닫힌 로그 검증을 유지한다. 준비 파일의 존재만으로 실행 중이라고 판단하지 않는다.

**10/07 05:31 KST에 GPU3의 별도 비교 실행을 시작했다.** Code `c3f445f`를
main에 push하고 입력8개의 기존 Drive 크기·MD5 검증을 마친 뒤 실행했다.
실제 writer438071의 소유자·고유 run·`CUDA_VISIBLE_DEVICES=3`을 확인했다.
확인 시점은 초기 장면 준비 단계이며 새 Q·actor 갱신과 전체 초기 평가는 아직
확인 전이다. 별도 TRAIN1,536조건과 원래 DEV128요청, 독립 FINAL 미사용을
유지한다. 기존 writer2701747·3847594도 유지했고 어떤 프로세스에도 signal을
보내지 않았다. [실행·입력 백업 근거](assets/rl_v2_servo_critic_actual_start_20261007.json).

**이후 실제 초기 DEV에 진입했다.** 진행 JSON에서 새 servo Q 계약과 성공 TRAIN
15궤적/6,899행, 같은 크기의 n-step 은행, 평가 전이0개를 확인했다. Actor·Q 갱신과
온라인 replay는0이므로 새 학습의 개선을 뜻하지 않는다.
[실제 평가 진입·Q 계약·은행 연결](assets/rl_v2_servo_critic_first_actual_DEV_20261007.json).

**학습 전 전체 초기 평가는23/128(중간 좌12·우6, 상단 좌5·우0)**으로 끝났다.
나머지는 랙 충돌69·시간 초과25·초기화 무효11건이며 무효도 원래128개 분모에
포함했다. 실제 안전한 양손 접촉·유지·들기23건과 actor0·Q0를 확인하고 정확한
초기 체크포인트를 별도로 보존했다. 옮겨온 초기 정책의 기준이며 새 학습의 개선은
아니다. 이후 첫 TRAIN에 진입했고 학습 후 전체 평가는 **자신의 초기23건**과 비교한다.
같은 배치 요청이어도 다른 실행의 flap 추첨·물리 이력까지 같다고 가정하지 않는다.
[전체 초기 평가와 일치 체크포인트](assets/rl_v2_servo_critic_full_initial_DEV_20261007.json).

**06:13 KST에 실제 Q418회 갱신·새 TRAIN24,503전이를 확인했다.** One-step Q와
실제 n-step16 보조 손실이 모두 유한했고 Q 표본256개 중 성공 TRAIN51개, 별도
16스텝 표본64개·가중치0.1이 연결됐다. 성공·n-step 은행은 초기15궤적/6,899행이며
초기 DEV의23개 성공은 넣지 않았다. 진행 중 VR·teacher BC 가중치는0이다.
Q는 replay64행부터 갱신하지만 actor는 Q2,048회·실제 replay32,768행을 기다린다.
이 시점에는 actor0회, Q warmup1,630회와 수집8,265행이 남았으며 아직 학습 후
파지 성능을 평가하지 않았다. 새로 갱신한 모델의 일치 checkpoint·전체 tensor
검증도 이 첫 진행 JSON 확인에 포함하지 않는다.
[실제 TRAIN·새 Q 연결](assets/rl_v2_servo_critic_first_real_TRAIN_Q_20261007.json).

**첫 갱신 모델도 따로 보존해 확인했다.** Q1,024회·actor0회의 체크포인트에서
Q·target Q 각각10개 tensor의 변화와 모든 모델 값의 유한성을 확인했다.
Actor와 actor 정규화는 초기 모델과 정확히 같았다. 같은 모델을 엄격하게 복원해
실제 성공 TRAIN15개에서 파지 구간480행을 읽기 전용으로 검사했다.
그중 서로 다른 유효 목표가 **같은 전체 모터 명령**을 만드는262행에서는
두 학습된 Q가 정확히 같았고, 포화된699개 몸체 좌표의 목표-Q 경사는0이었다.
이는 같은 상태에서 한 스텝 명령에 대한 불변성이다. 이후 전체 궤적이 같거나
새 정책이 더 잘 파지한다는 검증은 아니다. 활성 HDF·replay나 optimizer를 건드리지 않았다.
[실제 갱신 모델·명령 불변성](assets/rl_v2_servo_critic_Q1024_actual_model_20261007.json).

**이후 Q warmup을 통과해 actor도 실제로 갱신하기 시작했다.** 두 번째 TRAIN에서
actor26회·Q2,152회·실제 TRAIN100,861행과 유한한 actor·one-step Q·16스텝 Q 손실을
확인했다. 성공 보조 표본64개 중 파지 직전 지정 표본32개가 사용됐고,
성공 은행은 새 TRAIN 성공을 포함해20궤적/9,311행이 됐다. 새 실패도16스텝
은행에 수집하며 두 은행의 평가 전이는0이다. 진행 중 VR·teacher BC는0이다.
이는 정책 학습이 실제로 시작됐다는 근거다. 첫 학습 후 DEV128에서 자신의 초기
23건을 넘는지와 상단 양쪽 성공을 확인하기 전까지 성능 개선으로 세지 않는다.
[실제 actor 갱신·64/32 표본](assets/rl_v2_servo_critic_first_actual_actor64_tail32_20261007.json).

기존 tail 표본 비교의 첫 학습 후 DEV는 **23→14/128(중간 좌9·우5, 상단 양쪽0)**이었다.
모든128조건이 유효했고 랙 충돌85·과도한 들기2·시간 초과27건이었다. 파지 직전 구간을
더 자주 학습한 것만으로 초기 성능을 넘지 못했다. 새 Q 입력 비교와 구분해서 기록하며
성공·무작위화·충돌 기준을 완화하지 않는다.
[기존 tail의 첫 학습 후 전체 평가](assets/rl_v2_success_actor_tail_first_full_DEV_after384_20261007.json).

코드: [encoder](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/servo_critic.py),
[새 pilot](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/actual_flap_servo_critic_sac.py),
[공통 hybrid Q 배선](../src/kuavo_isaaclab_scene/rl/algorithms/hybrid_goal_sac.py),
[초기화 도구](../scripts/rl/prepare_servo_critic_actor.py).
