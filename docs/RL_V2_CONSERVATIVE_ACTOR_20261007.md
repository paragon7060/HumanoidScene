# 10/07 SAC actor 갱신 폭 비교

누적 보상을 보강한 SAC의 첫 전체 평가는27→8/128로 떨어졌다. 같은 과거의
실제 성공 TRAIN 관측에서 학습 후 actor706의 제어 명령 오차가 커지고, 상단
왼쪽의 양쪽 닫힘 명령도217개 관측 중34개로 줄었다. 이는 성공 동작을 잃는
흔적이며 모든 실패의 원인을 단독으로 증명하는 분석은 아니다.
[전체 실제 평가](assets/rl_v2_episode_return33_first_full_DEV_20261007.json),
[같은 성공 관측의 명령 변화](assets/rl_v2_episode_return33_actor706_same_TRAIN_diagnostic_20261007.json).

실제 저장된 optimizer의 학습률은 **actor1e-5, Q·연속/이산 entropy3e-4**였다.
actor도3e-4라고 가정해 조정하지 않는다. 이번 선택형 비교는 몸체19개·양손
그리퍼2개의 공유 actor optimizer만 **1e-5→1e-6**으로 낮춘다. Q·두 target·
entropy 학습률과 모든 학습 손실, actor 갱신 간격·warmup은 유지한다.

시작 정책은 최고33/128 모델에서 보존한 동작과 Gaussian이다. 현재 정책을
TRAIN의80%, 기존 탐색을20% 사용하는 비교의 **같은 초기 모델과 동일한
TRAIN1,536/DEV128 요청 목록**을 사용한다. 새로운 Q와 빈4개 optimizer·
온라인 replay·성공 은행·누적 보상 은행에서 시작한다. 기존 보상 전이나 평가
경로는 가져오지 않는다. 성공·실패의 실제 할인 누적 보상 보강은 유지한다.

박스·base·배경·움직이는 flap의 기존 무작위화를 유지한다. 양손 실제 pad 접촉,
0.25초 유지와8mm 들기, rack10N·주변 장애물5N·self collision OFF의 성공·
안전 조건도 유지한다. Curriculum이나 박스 고정은 추가하지 않는다. 같은
요청 목록이어도 물리 reset·flap 추첨이 동일하다고 가정하지 않는다.

관련32개 검사를 통과했다. 같은 실제 hybrid SAC 업데이트에서 actor의
파라미터 변화는 약1/10이었고, Q·target 업데이트와 학습 손실은 정확히 같았다.
학습률이 다른 optimizer를 잘못 복원하거나 학습·수집을 시작한 초기 입력을
새 설정으로 옮기는 경우는 거부한다. 별도 artifact type으로 기존 설정을 보존한다.

CPU에서 **전체 trainer를 실제로 복원**해 과거 자기 TRAIN1,165관측의 몸체·
그리퍼 동작과 Gaussian을 확인했다. 처음 모델의 모든 tensor는 동일했고 Q·
entropy optimizer도 그대로였다. 새 정책의 Q 영상 복원에서도 모든 모델
tensor가 정확히 같았다. 이는 초기 연결·학습 갱신 폭 확인이며 파지 성능
향상을 뜻하지 않는다. [실제 복원 근거](assets/rl_v2_return33_conservative_fulltrainer_preparation_20261007.json).

고유 실행 폴더와`CUDA_VISIBLE_DEVICES=3`을 사용하고 기존 비교는 유지한다.
기존 Drive 연결로 초기 체크포인트·계약7개를 검증하며, 학습 중에는300초마다
체크포인트를 검증하고 최신2개를 보존한다. 종료 후 닫힌 로그를 검증한다.
Raw replay/HDF는 RAM 실행 폴더에 보존하며 업로드·자동 삭제 대상으로 삼지 않는다.
실제 실행과 첫 학습 후 전체128개 평가를 이후 근거로 구분해 기록한다.

**19:37 KST에 실제로 시작했다.** 소스`1aec17a`, 실행
`batch_sac_20261007_193730_d5822a`의 고유 관리 폴더에서 writer454955의
소유자·실행 명령·CUDA3와 관리자·서비스를 확인했다. 기존6개 writer는 유지했다.
초기 입력7개의 기존 Drive 크기·MD5 검증도 완료했다.
[실제 실행·백업 범위](assets/rl_v2_return33_conservative_actual_launch_20261007.json).

이후 실제 첫 초기 DEV에 진입했다. Actor 학습률1e-6·Q 학습률3e-4,
actor518/critic578·초기 agent/progress/입력 계약의 정확한 일치와 학습 카운터0을
확인했다. 이후 전체 초기 평가는 **27/128(중간 좌16·우11, 상단 양쪽0)**으로
끝났다. 유효117·초기 무효11건을 원래128개 분모에 포함했다. 기존 순서대로
초기 평가 뒤 저장한 체크포인트0은 Drive 검증 후 로컬 보존 규칙으로 정리됐다.
기존 연결에서 이 실행의 모델 한 개만 RAM에 회수해 크기·MD5를 재확인했고,
모든 모델·정규화 tensor가 준비 입력과 정확히 같고 유한함을 확인해 보호했다.
초기27건은 학습 개선이 아니다. 첫 학습 후 전체 평가는 아직 확인 전이며 CPU
전용 관찰자가 그 DEV4와 같은 모델을 보호한다.
[실제 첫 평가의 학습률·계약](assets/rl_v2_return33_conservative_first_actual_DEV_20261007.json).
[완료된 초기 평가·동일 모델](assets/rl_v2_return33_conservative_full_initial_DEV_20261007.json).

준비 명령:

```bash
python scripts/rl/prepare_conservative_servo_actor.py \
  --initial-checkpoint /absolute/path/to/pristine_init/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/pristine_init/training_manifest.json \
  --waypoints /absolute/path/to/pristine_init/waypoints.json \
  --output-dir /absolute/path/to/unique_conservative_init
```

생성된 체크포인트의 artifact type으로 배치 학습·정책 재생이 같은 클래스를
선택한다. 별도 환경 보상 옵션이나 기존 실행의 optimizer 수정은 필요하지 않다.
독립 FINAL·다른 박스 크기·네 구역에서 안정적인 성공은 아직 입증되지 않았다.

다음 데이터 후보도 읽기 전용으로 확인했다. 이전 SAC의 실제 TRAIN 탐색에서
성공한 상단 오른쪽1경로·590개 관측의 몸체 명령은 현재 actor의 출력 가능
범위에 모두 들어왔고 닫힘 명령도 현재 jaw gate를 모두 통과했다. Actor 관측,
목표 좌표·flap 인지 계약과 고정 몸체 anchor가 일치했다. 과거의 안전한 성공
동작을 잃지 않는 actor 전용 기억 후보이며, **현재 실행에는 넣지 않았고 기존
Q·보상 전이도 가져오지 않았다.** 실제 환경·성공 기준과 TRAIN/DEV 분리의
최종 호환성 확인, 별도 actor 전용 데이터 연결과 새 전체 평가는 아직 필요하다.
[읽기 전용 TRAIN 명령 호환성](assets/rl_v2_actual_TRAIN_upper_right_actor_memory_eligibility_20261007.json).
