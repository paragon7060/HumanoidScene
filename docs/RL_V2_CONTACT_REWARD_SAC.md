# V2 grasp: 실제 접촉 진행을 보강하는 SAC

2026-10-05, frozen 파지 구간 진단을 바탕으로 선택적 보상 프로필
`opposing_pad_contact_progress_v1`을 추가했다. 기본 v2 보상은 그대로다.
변경된 보상으로 기존 Q/replay를 재개하지 않고, 기존 actor의 실제 동작만
보존한 새 SAC를 GPU0에서 시작했다. 기존 GPU3 실행은 유지한다.

![접촉 품질과 보상 가중치](assets/rl_v2_contact_reward_sac_20261005.png)

그림 왼쪽은 수식의 예시이며 측정 성공률이 아니다. 오른쪽은 설정한 가중치다.
[입력·초기 동작 대조·실행 증거](assets/rl_v2_contact_reward_sac_20261005.json).

## 진단에서 확인한 문제

[전체 파지 구간 진단](RL_V2_PHYSICAL_BODY_SAC.md)은
기존 DEV 사례 중 성공했던9개와 가까이 접근한 실패7개를 고른 cold N16 실행이다.
16개 중14개는 양손이 flap 표면3cm 이내로 접근했지만, 양손 실제 pinch는5개,
lift/hold 성공은3개였다. 위쪽 왼쪽4개는 모두 가까워졌지만 양손 pinch가 없었다.
대부분의 실패에서는 실제 pad 접촉이 없거나 한쪽 pad의 힘이5N에 미달했다.
선택한 사례의 점수를 전체 일반화 성공률로 해석하지 않는다.

Actor/reward 목표 ID와 hand–flap relation은 일치했다. 전체 close-gate 판정 중
nominal/actual flap 차이가 결정을 바꾼 비율은0.22%였다. 이 증거에서는 목표 입력
오배정보다 접근 이후 양쪽 pad의 접촉 형성과 유지가 다음 수정의 우선순위다.

GPU3의 동일 DEV128 재개 결과는9→8→11 성공이다. 최신 영역별 성공은
중간왼쪽4/32, 중간오른쪽6/32, 위왼쪽1/32, 위오른쪽0/32였다.
아직 안정적인 네 영역 성공 또는 통계적으로 확인한 개선이 아니다.

## 접촉 보상 계산

기존 filtered finger–flap force, pad 접촉영역, jaw 대향 판정을 재사용한다.
새 sensor나 physics substep은 추가하지 않는다. 각 pad의 값은
`p = clamp(force_N / 5, 0, 1)`이며 실제 flap 접촉영역 밖이면0이다.
각 손의 같은 flap 양쪽 pad 품질은 다음과 같다.

```text
hand_quality = 0.25 * (p_front + p_back) + 0.5 * min(p_front, p_back)
```

Jaw가 대향하지 않으면 해당 hand/flap 값은0이다. 서로 다른 flap에 양손을
배정하는 두 조합 각각에 대해 다음 값을 계산하고 큰 값을 선택한다.

```text
phi = max_over_two_opposing_assignments(
    0.25 * (q_left + q_right) + 0.5 * min(q_left, q_right)
)
contact_quality_progress = 1.0 * (0.999 * phi_next - phi_previous)
```

한 손의 pad 하나만5N이면 phi=0.0625, 한 손의 양 pad만5N이면0.25,
양손이 서로 다른 flap의 양 pad를 모두5N으로 집으면1이다. 두 손이 같은 flap을
집으면1이 되지 않는다. 5N보다 강하게 누른다고 값이 커지지 않는다.
비유한 힘, 사용할 수 없는 센서, 같은 손의 flap 판정이 모호한 경우는0이다.
거리나 닫기 명령만으로 접촉 값을 만들지 않는다.

보상은 접촉 품질 자체를 매 step 더하는 방식이 아니라 potential 차이다.
잡았다 놓으면 감소분이 돌아오고, 같은 접촉을 유지하면 작은 discount 비용이
남는다. 성공·안전 실패·timeout은 absorbing potential0으로 처리해 반복 접촉이나
timeout 직전 접촉으로 보상을 쌓는 것을 막는다. Reset/settling 후 첫 유효 관측은
기준만 설정하며 초기 접촉 보너스를 주지 않는다. 실제 최종 성공 보상은 별도다.

환경의 원래 `terminated`/`truncated` 기록은 유지한다. 이 프로필에서는 timeout을
Q의 absorbing 상태로 취급해 `Q_done = terminated | truncated`를 사용한다.
따라서 timeout potential0과 bootstrap 규칙이 일치한다. 기존 보상의 SAC는
원래 timeout bootstrap을 유지한다. 새 의미는 보상 계약에 명시되며 변경 전
Q/replay를 이어 쓰지 않는다. 이 연결 수정 전에 첫 GPU0 DEV 실행을 정상 종료했고
actor/Q 업데이트는 모두0이었다. 수정한 계약으로 별도 초기 폴더와 실행을 만든다.

새 manifest에 남아 있는 기본 `terminal_contract.timeouts_bootstrap=true` 설명보다
`reward_profile.contact_shaping.time_limit_is_absorbing_for_Q=true`의 명시적 규칙이
우선한다. 아래 요약 도구는 두 값을 따로 표시하며 실제 새 Q의 timeout bootstrap은
false다. 설명 필드만 보고 기존 Q 의미로 해석하지 않는다.

## 변경한 가중치

| 항목 | 기존 | 접촉 프로필 |
|---|---:|---:|
| 접촉 품질 진행 | 없음 | 1.0 |
| 한손 pinch event | 2.0 | 0.5 |
| 양손 pinch event | 1.0 | 2.0 |
| 실제 lift/hold 성공 event | 5.0 | 8.0 |
| box drop·비정상 box 실패 비용 | 8.0 | 12.0 |
| workspace limit 비용 | 8.0 | 12.0 |

한손만 잡는 event가 양손 event보다 컸던 비율을 바꿨다. 성공8은 기존 양의
기하 potential 가중치 합4.4와 새 접촉1보다 크다. 파괴적 종료 비용12는 성공8보다
크게 유지한다. 프로필의 high-level failure도8→12로 맞추지만 현재 grasp runner가
high-level task 보상을 추가 지급하는 것은 아니다.

접근2, front staging1, 정렬0.5, capture0.5, proof lift0.4, jaw gap0과 기존
거리/action 비용은 유지한다. 성공 조건·rack10N·robot-only obstacle5N·self-collision
OFF·floor/target box 제외·finger–rack 검출을 유지한다. 박스는 동적이며 base XY/yaw,
박스와 주변 박스 randomization, 중간/위 좌우 네 영역을 유지한다.

## 새 보상에 맞는 학습 상태

`scripts/rl/prepare_contact_reward_sac.py`는 matching nominal hybrid held-goal
checkpoint에서 actor와 actor normalizer를 복사한다. 새로운 보상임을 명시적으로
기록하고 Q/target-Q/critic normalizer/entropy/모든 optimizer/학습 counter/replay 및
TRAIN 성공 저장소는 새로 시작한다. 이전 보상의 성공 전이를 다시 라벨링하지 않는다.
관측·action·waypoint·물리·종료 계약은 그대로 비교한다. 알려진 보상 변경을 무시하는
계약 변환은 frozen 기반 actor를 읽을 때만 사용하며 Q/replay 호환 검사를 우회하지 않는다.

준비 중 실제 초기화 문제를 발견했다. Categorical jaw logit은 원래 frozen network와
현재 actor의 차이를 사용한다. Actor tensor만 복사하고 이 기준 network를 learned
actor로 바꾸면 실행되는 열기/닫기가 달라진다. 첫 준비본에서는 실제 TRAIN 입력498개
중22개 손 명령이 달랐고 GPU 학습을 시작하지 않았다. 원래 jaw 기준 network와
normalizer도 보존한 수정본은498개 입력의 body/jaw 목표가 정확히 일치했다.
이것은 이번 준비 과정의 오류이며 이전 학습 전체의 원인이라고 주장하지 않는다.

Source actor8516의 prior fade 진행7170회를 보존해, Q counter를0으로 초기화해도
이미 끝난 기존 BC-prior loss를 다시 켜지 않는다. TRAIN 성공 유지 항은 새 보상으로
실제로 모은 성공에서 다시 시작한다. 기존 탐색 설정(std0.001, 범위0.001..0.005,
AR(1)0.98, goal 반경0.05, jaw confidence0.8/residual gain20), replay50만행,
critic warmup2048회, actor 최소 수집32768행을 유지한다.

```bash
conda activate env_isaaclab_232
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  python scripts/rl/prepare_contact_reward_sac.py \
  --checkpoint /absolute/path/to/matching-legacy-held-hybrid-checkpoint.pt \
  --training-manifest /absolute/path/to/matching-legacy-training-manifest.json \
  --waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --native-seed /absolute/path/to/current-middle-calibration.hdf5 \
  --native-seed /absolute/path/to/current-upper-calibration.hdf5 \
  --output-dir /absolute/path/to/unique-contact-initial
```

기존 native 입력2개는 frozen 기반 actor의 물리 계약 대조용이다. 새 Q seed가 아니다.
Source에 실제 TRAIN actor 입력이 없으면 초기 동작 검증을 통과시키지 않는다.

## 실행·평가·보관

새 초기 폴더의 `training_manifest.json`와 `checkpoint_00000000.pt`를 함께 사용한다.
관측 contract나 물리 설정을 임의로 수정한 manifest로 재개하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl/batched_staged_goal_with_drive.py \
  --gpu 0 --training \
  --experiment-dir /absolute/path/to/unique-contact-training-run \
  --checkpoint /absolute/path/to/unique-contact-initial/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/unique-contact-initial/training_manifest.json \
  --waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --demo-dataset /absolute/path/to/HumanoidScene/examples/demos/v2_grasp_quest_success.hdf5 \
  --native-seed /absolute/path/to/current-middle-calibration.hdf5 \
  --native-seed /absolute/path/to/current-upper-calibration.hdf5 \
  --waves-json /absolute/path/to/balanced-TRAIN-DEV-waves.json \
  --stop-on-validation-regression --minimum-validation-region-success-rate 0 \
  --validation-regression-significance 0.05
```

이번 실행은 기존 N128 wave 목록의 첫13개(8TRAIN/5DEV)만 사용한다. 최종 wave도
DEV이며 독립FINAL은 포함하지 않는다. 배치와 randomization 범위를 줄인 curriculum이
아니라 저장공간·성과를 검토할 첫 학습 블록이다. 이후에는 같은 새 보상 checkpoint와
동일 실행의 replay로 이어간다. 처음13 wave 종료를 전체 목표 달성으로 취급하지 않는다.

`metrics.json`의 outcome에는 실제 reset 직전 `contact_quality`를 기록한다.
값1만으로 lift/hold 성공을 선언하지 않으며 기존 success/pinch/unsafe/hold/lift 필드를
함께 본다. 실제 reward와 breakdown 합 일치 검사는 학습 중 계속 실행한다.
같은 matching manifest로 `--no-training` DEV wave를 실행하면 모델/replay를 고정한다.
단일 영상 `replay_v2_grasp_reference.py --staged-goal-sac`도 이 프로필을 명시적으로
읽고 같은 보상 manager를 설정한다. 이번 변경에서 단일 영상의 GPU 재생 검증은 별도다.

기존 [Drive 저장 안내](RL_GOOGLE_DRIVE.md)에 따라 같은 인증과 dedicated folder를
재사용한다. 초기 model/replay/manifest/검증 자료의 크기·MD5를 확인하고 시작했다.
관리자는 checkpoint를300초마다 검증 업로드하고 최신 두 개를 보호한다.
열린 HDF/log는 종료 후 닫힌 상태에서 최종 업로드한다. Replay 원자 저장의 임시
복사본과 열린 HDF는 여전히 로컬 공간이 필요하므로 Drive를 mounted disk로 취급하지 않는다.

접촉 점수·모호한 sensor·힘 포화·접촉/timeout 보상 순환·partial reset·엄격한 계약·
fresh Q migration·categorical jaw 기준 network 보존 등 관련 CPU34개 검사가 통과했다.
새 reward의 실제 학습 효과는 이후 같은 DEV128에서 검토해야 하며, 초기화 통과나
첫 Q 업데이트 자체가 파지 성공을 뜻하지 않는다.

## 18:56: 실제 Q 학습 시작 확인

수정한 GPU0 실행의 첫 frozen DEV는6/128(ML1/MR3/UL1/UR1)이었다.
Actor/Q 업데이트는 모두0이며 새 보상의 학습 효과가 아니다. 초기 유효 배치는99/128이고
실패한 reset도 전체128회 분모에 남는다. Cold 물리 초기화에 따른 변동을 고려한다.

이어 첫 TRAIN의91step에서 critic56회 업데이트를 확인했다. Actor는0회로,
critic2048회 warmup과 실제 replay32768행 조건을 기다린다. 원래 입력498개의
초기 body/jaw 동작 보존, 실제 reward와 breakdown 합 일치, GPU0 격리를 확인했다.
로그의 actual_rows는 DEV와 base 접근도 포함하므로 TRAIN replay 크기로 해석하지 않는다.
실행 소스의SHA256도 공개 commit78fbd63의 바이트와 일치한다.

기존 GPU3의 다음 DEV는6/128로,9→8→11→6의 변동을 보여 지속 개선을 확인하지
못했다. 새 GPU0의 변경 후 DEV는 아직 없고 네 영역 일반화 목표는 완료되지 않았다.

닫힌 DEV 진단9827행에 접촉 수식만 적용한 조건부 preview는16개 중14개에 신호를
만들었다(기존 양손 pinch5개). 센서 availability가 해당 trace에 없으므로 이 preview는
availability=True를 가정한 수식 점검이다. 실제 reward 측정이나 새 Q seed가 아니고,
기존 reward를 relabel하지 않았다. 실제 새 학습에서는 availability를 직접 검사한다.

## 19:40: 첫 actor 학습 후 DEV 결과

![첫 학습과 DEV 결과](assets/rl_v2_contact_reward_first_learning_20261005.png)

[측정 snapshot·실제 성공 전이 Q 진단](assets/rl_v2_contact_reward_first_learning_20261005.json).
GPU0의 첫 두 TRAIN에서 실제 양손 lift/hold 성공12건을 수집했다. 새 보상의
성공 저장소5308행은 ML2/MR8/UL2/UR0 에피소드이며 DEV 전이는0행이다.
Actor230회, critic2966회, 실제 held TRAIN replay99667행이다. 기존 BC-prior loss는
0이고 성공 actor 보조 항은 이번 TRAIN에서 모은 실제 경험에만 적용한다.

첫 학습 후 DEV는 **6/128→6/128**으로 전체 개선이 없다.

| 영역 | 초기 DEV | actor230/Q2966 후 DEV | 두 실행 모두 초기 유효한 사례 |
|---|---:|---:|---:|
| 중간왼쪽 | 1/32 | 2/32 | 18개에서1→2 |
| 중간오른쪽 | 3/32 | 3/32 | 23개에서2→3 |
| 위왼쪽 | 1/32 | 1/32 | 30개에서1→1 |
| 위오른쪽 | 1/32 | 0/32 | 25개에서1→0 |

모든128개는 같은 requested layout이다. Initial valid는99→98이며 공통 유효96개에서는
5→6 성공이다. 공통 유효만의 비교는 진단용이고 원래 성공률 분모128을 바꾸지 않는다.
반복 reset의 실제 물리 이력까지 같다고 주장하지 않는다. 기록된 성공6건 모두
양손 pinch·opposing flap·안전·0.25초 hold·8mm proof lift를 충족했고, unsupported
success flag는 없었다. 위오른쪽 감소는 기록하되 현재 회귀 guard의 exact paired
검사에서는 유의한 손실이 아니었다. 다음 TRAIN wave를 같은 Q/replay로 진행한다.

Q2966 checkpoint의 model tensor는 모두 finite였다. 실제 TRAIN 성공12건의
terminal pre-state와 실행된 action만 CPU로 읽어 계산한 평균 min(Q1,Q2)는4.15,
실제 terminal target은6.99였다. 성공의 양의 신호가 새 critic에 연결되어 있으며
아직 약2.84 낮게 평가한다. 이 값은 성공률이나 독립 일반화 성능이 아니다.

첫 post-TRAIN DEV의 초기 유효98건 중59건은 unsafe,33건은 timeout이었다.
Unsafe 원인에는 rack29건, box speed32건, lift limit12건, workspace6건,
drop6건, obstacle1건이 있었다. 한 실패에 여러 원인이 겹칠 수 있어 이 숫자를
서로 더해 실패 수로 쓰지 않는다. 같은 DEV에서 종료 직전 양손 pinch는6건이었다.

비정상 속도/위치로 실패한 박스의 마지막 flap distance에는 극단적으로 큰 유한값도
있다. 이를 안전한 접근 거리의 평균으로 해석하면 지표가 왜곡된다. 실패를 삭제하지
않고 종료 결과별 거리를 분리한다. Safe timeout33건의 마지막 손–flap surface 거리
중앙값은 왼손4.37cm/오른손4.06cm였다. 이는 마지막 상태이고 episode 중 최고
접촉/접근 점수가 아니다. 초기화 실패와 unsafe 종료도 계속 해결할 항목으로 남는다.

같은 시각 기존 GPU3 DEV 이력은9→8→11→6→10/128이었다. 최신 영역별 성공은
ML1/MR8/UL1/UR0으로 중간오른쪽에 편중되고 네 영역의 안정적인 개선은 없다.
두 실행은 시작 Q/replay와 reset 이력이 달라 보상 변경의 인과 비교로 쓰지 않는다.
두 실행 모두 독립FINAL 기록은0개이고 네 영역 일반화 목표는 진행 중이다.

## GPU 없이 완료된 학습 지표 읽기

표준 Python만으로 runner가 저장한 완료 wave를 요약한다. Isaac/Torch/GPU/Drive를
사용하지 않고 학습·replay·optimizer를 변경하지 않는다.

```bash
python3 scripts/rl/summarize_batched_staged_run.py \
  --run-dir /absolute/path/to/batch_sac_run \
  --output /absolute/path/to/new-progress-snapshot.json
```

`--output`은 기존 파일을 덮어쓰지 않으므로 매번 새 경로를 지정한다. 생략하면
표준 출력으로 표시한다. `recorded_writer_status`는 저장된 상태일 뿐 현재 PID가
살아 있다는 증거가 아니다. 실행 여부는 관리 폴더의 PID와 실제 프로세스를 별도로
확인한다. JSON을 쓰는 순간의 불완전한 내용은 잠깐 다시 읽고, 계속 실패하면
관측 오류를 보고하며 학습이 중단됐다고 추정하지 않는다.

- `held_TRAIN_rows_total`/`current_TRAIN_replay_rows`: 실제 Q에 사용되는 수집량/현재 buffer.
  `last_recorded_collection_rows_all_splits`에는 DEV와 base 접근도 포함한다.
- `waves`: 영역별 전체 시도·초기 유효·성공 조건 대조·unsafe·timeout, 겹치는
  개별 안전 원인과 동시 원인 조합, 종료 직전 접촉과 결과별 거리 분포.
- `paired_DEV_against_initial`: 같은 requested layout인지와 공통 유효 사례의 변화.
  초기화 실패를 원래 분모에서 제외하는 새 점수가 아니다.
- `effective_Q_timeouts_bootstrap`: 보상 프로필의 우선 규칙을 적용한 실제 Q 의미.
- `recorded_FINAL_waves`: 이미 기록된 독립FINAL 수. 계획된 미래 평가 수가 아니다.

이번 지표 도구는 GPU0/3의 실제 완료 snapshot을 읽어 결과를 대조하고 Python
compile을 확인했다. 학습 코드와 활성 writer의 설정을 다시 바꾸지는 않았다.

## 20:15: 두 번째 학습 후 DEV와 actor 영향 점검

![실제 actor 변화와 두 번째 학습 후 DEV](assets/rl_v2_contact_reward_actor_effect_20261005.png)

[실제 TRAIN 입력의 CPU 진단·완료 DEV·후속 입력 증거](assets/rl_v2_contact_reward_actor_effect_20261005.json).
GPU0 actor986/Q5990에서 고정한 다음 DEV는5/128(ML0/MR4/UL1/UR0)이었다.
기존6→6→5로 개선이 없다. 초기 유효98개 중 성공5/unsafe63/timeout30이고
초기화 실패30개도 원래 분모128에 남는다. Rack 충돌45, box speed21, lift limit10,
drop4, workspace3은 중복 가능한 원인 수다. 마지막 양손 pinch6건 중 실제 성공은5건이다.
네 TRAIN의 실제 성공은 총23건이지만 위오른쪽 TRAIN 성공은0건이다. 저장소는
지역별 용량 때문에 오래된 성공을 제거하므로 누적 성공 건수와 같지 않다.
해당 시각 저장소에는16에피소드/6949행(ML5/MR9/UL2/UR0), 실제 TRAIN replay198232행이
있고 평가 전이는0행이다. 기존 GPU3의 다음 DEV도10/128(ML3/MR7/UL0/UR0)으로
9→8→11→6→10→10이며 상단 성공이 없어 지속적인 네 영역 개선은 없다.

새 Q5990와 원래 초기 actor0를 같은 실제 TRAIN 입력으로 CPU에서 대조했다.
무작위4096행의 실행되는 normalized body 목표는 평균 절대값0.00837만큼 바뀌었고,
gripper의 deterministic 결정35개가 달랐다. Body 좌표의 약10.0%는 projector 경계에서
Q gradient가0이며 나머지는 Q→목표 gradient가 통과했다. 성공 경로354행에서는
차단 비율0.58%였다. 관측 입력·정규화·원래 frozen jaw 기준을 그대로 복원했으며
optimizer/replay를 수정하지 않았다. 이것은 실제 몸의 안전한 동작을 보증하는 검사가 아니다.

안전한 TRAIN 성공16건의 마지막 실제 state/action에서 평균 minQ6.83, target6.99,
MAE0.40/RMSE0.56이었다. 평균만 같다고 완전히 수렴했다고 주장하지 않는다.
성공의 보상 신호가 연결되고 actor 출력도 바뀌지만, 아직 실제 DEV 성공률은 개선되지
않았다. 따라서 학습이 전혀 작동하지 않거나 projector가 모든 움직임을 막는다는
설명만으로 현재 결과를 설명할 수 없다. Box/접촉 물리와 실패하는 손의 geometry를
함께 확인하며 같은 학습을 이어간다.

첫 학습 블록 이후 사용할 원래 wave12..24(8TRAIN/5DEV/FINAL0)를 준비했다.
각 wave128개/구역32개, 원래 layout과 DR을 정확히 보존한다. 현재 writer가 종료한 뒤
같은 새 보상 checkpoint와 그 실행의 실제 Q/replay를 확인해 재개한다. 아직 새 학습을
중복 실행한 것은 아니다.

### 원래 DEV128 전체를 한 장면씩 진단

병렬 물리 환경의 영향과 실제 flap/pad geometry를 확인하기 위해, 같은 actor986/Q5990를
고정해 원래 DEV128 전체를 N1 장면128회로 재생할 입력을 준비하고 Drive 크기/MD5를
검증했다. 사례를 골라 줄이지 않으며 box/base/background DR, 중립 reset, 성공·안전
조건과 물리 설정을 유지한다. N1의 world 구성과 물리 이력은 N128과 다르므로 같은
초기 상태의 인과 비교나 N128 성능 향상으로 해석하지 않는다. 결과는 실제 outcome으로
따로 확인해야 한다. 기존 두 학습은 계속 진행한다.

`--grasp-observation-audit --full-distribution-grasp-observation-audit --no-training`은
서로 다른128개 DEV seed를128개 병렬 환경의 한 wave 또는 한 환경의128개 wave로
실행하면서 실제/nominal flap 자세, pad 힘·영역,
closing gate와 움직임을 기록한다. 4영역 각32개, original 주변 배치만 허용한다.
TRAIN/독립FINAL, 중복 seed, 불균형/일부 사례만의 목록, 다른 physics probe와의 결합은
거부한다. 기존 한 wave/최대16개 선택 진단은 그대로다. 모든 진단 전이는 Q/replay나
성공 actor 보조 손실의 입력으로 사용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl/batched_staged_goal_with_drive.py \
  --gpu 0 --experiment-dir /absolute/path/to/unique-full-DEV128-N1-run \
  --checkpoint /absolute/path/to/verified-frozen-input/checkpoint_00005990.pt \
  --training-manifest /absolute/path/to/verified-frozen-input/training_manifest.json \
  --waves-json /absolute/path/to/verified-frozen-input/waves.json \
  --waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --demo-dataset /absolute/path/to/HumanoidScene/examples/demos/v2_grasp_quest_success.hdf5 \
  --native-seed /absolute/path/to/current-middle-calibration.hdf5 \
  --native-seed /absolute/path/to/current-upper-calibration.hdf5 \
  --no-training --grasp-observation-audit --full-distribution-grasp-observation-audit
```

원래 layout의 내부 split은 legacy `holdout`이고 wave split은`validation`이다.
독립FINAL은 별도 outer`holdout` wave 및 겹치지 않는 seed 집합으로 구분한다.
기존 row를 새 split으로 다시 라벨링하지 않는다. Wave마다 audit packet/중복 검사를
새로 시작하고 각 기록에 원래 wave/seed/region을 붙여 같은 local step의 다른 사례가
누락되지 않게 한다. 위 guard와 접촉 보상 관련 CPU32개
검사가 통과했고 독립FINAL을 학습/진단에 가져오지 않는다.

## 10/05 21:10 — 전체128개 병렬 접촉 진단과 단일 환경 비교

GPU3 legacy 학습의 DEV 성공 개수는 `9 → 8 → 11 → 6 → 10 → 10 → 9 /128`로,
지속적인 개선이 없다. 최신 DEV18은 중간 좌1/우7, 위 좌0/우1이다.
독립FINAL을 실행하기 전에 본인 소유 관리자에게 정상 종료를 요청했으며,
실제 GPU writer 종료와 GPU3 메모리 해제를 확인했다. 기존 관리자는 닫힌
checkpoint/replay/HDF/로그의 최종 Drive 검증을 계속 수행한다.
GPU0 새 접촉 보상 SAC는 계속 학습하고, DEV9에서 actor1740/Q9006을 평가한다.

전체 DEV128 N1 진단의 첫 사례(seed121000, 중간 왼쪽)는501 control step에서
양손의 서로 반대 flap 파지·pad force·안전·proof lift·hold 조건을 모두 충족했다.
최종 각 손의 pad 힘은 `[16.34,15.69] N`과 `[8.81,8.86] N`, hold0.267초,
rack clearance3.03cm다. 같은 요청 배치는 GPU0 N128 DEV0에서 성공했으나
DEV3/6에서는 랙 충돌로 실패했다. N1과 DEV6의 실제 rack world yaw, base pose,
settling 이력이 다르므로 이것만으로 병렬 물리 오류나 정책 개선을 주장할 수 없다.
단일 성공을 전체 분포 성공률로 사용하지 않는다.

모든128개 원본 DEV 사례를 유지한 N128 frozen audit를 추가할 수 있도록
진단 범위를 확장했다. N1과 똑같은 actor986/Q5990 checkpoint를 쓰고 실제/nominal
flap 중심·방향, 각 pad 힘·영역, closing gate, 종료 원인을 기록한다.
4영역 각32개, box/base/background DR, 기존 성공·안전·물리 설정을 유지한다.
TRAIN/독립FINAL, 일부 사례, 혼합 wave 구성, 중복 seed와 다른 물리 probe는
허용하지 않으며 진단 데이터를 학습 replay에 넣지 않는다. 기본 TRAIN 동작은 변하지 않는다.
병렬 입력은 원본 DEV wave와 정확히 같음을 확인하고 기존 Drive 연결로
크기·MD5까지 검증했다.

정상 종료 처리에서도 표시 오류를 확인했다. Batched runner는 요청된 종료를
`interrupted`로 쓰는데 이전 관리자는`stopped`만 인정해 exit0을1로 바꿨다.
관리자 자신의 종료 요청이 실제로 있었을 때만 두 상태를 인정하도록 수정했다.
요청 없이 불완전한 상태로 exit0을 반환하는 Kit 오류는 계속 실패로 판정한다.
이미 실행되던 GPU3 관리자의 exit1 기록은 그대로 보존하며, 정상 종료 요청과
실제 writer 종료 증거를 따로 기록한다. 이 수정은 새 관리자부터 적용된다.
접촉 진단·접촉 보상·실제 CPU 자식 프로세스 종료/백업 순서 검사55개가 통과했다.
