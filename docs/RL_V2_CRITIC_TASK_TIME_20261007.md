# SAC critic에 실제 남은 task 시간을 추가하는 옵션

종료 표본을 보강한 SAC도 시간 초과의 Q 오차가 남았다. 같은 실제 TRAIN17개
시간 초과 동작의 마지막 보상 평균은−2.413인데, 첫 모델의 Q는−1.269,
두 번째는+0.798이었다. [동일 표본 비교](RL_V2_TERMINAL_CRITIC_CREDIT_20261007.md).
이 오차가 시간 입력만의 문제라는 인과관계는 아직 확인하지 않았다.

## 발견한 시간 기준의 차이

기존 critic에는 파지 단계 시작 이후의 시간이 있었다. 실제 task timeout은
reset 정착을 통과한 뒤의 전체 시간에 적용되므로 base 접근에 쓴 시간도 포함한다.
같은 파지 진행 시간이라도 실제 남은 제한시간은 달라질 수 있다.

완료된 자기 TRAIN17개 시간 초과에서 전체 종료 step은 모두857이었지만,
파지 시작은54–97이었다. 기존 critic의 시간 값은0.843–0.891로 기록됐다.
**Critic의 시간 horizon은900이며, 이 표본에서 시간이 포화돼 사라진 문제는
아니다.** Actor의 제한된 진행 시간과 critic 시간을 혼동하지 않는다.
[실제 마지막 관측의 시간 대조](assets/rl_v2_actual_critic_held_clock_audit_20261007.json).

## 변경 내용

`--critic-episode-clock task-remaining`은 기존 파지 진행 시간을 유지하면서
아래 값을 critic에 한 개 더 입력한다.

```text
remaining = clamp(1 − reset_settling.ready_steps / env.max_episode_length, 0, 1)
```

종료 함수 `task_time_out()`과 **같은 실제 카운터·horizon**을 읽는다. 종료 뒤
자동 reset된 환경의 새 시간을 사용하지 않도록, 행동 전 관측과 reset 전
terminal 관측에서 각각 측정하고 HDF에 남긴다. 수치 오류로 일부 행이 제외돼도
남은 관측·행동·시간의 환경 번호를 같이 유지한다.

| 항목 | 기존 | 선택형 새 설정 |
| --- | --- | --- |
| Actor 입력 | 518 | 518, 그대로 |
| Critic 입력 | 577 | 578 |
| 기존 파지 시간 | index530 | 유지 |
| 실제 남은 task 시간 | 없음 | index571에 추가 |
| 마지막6개 held context | 있음 | 그대로 유지 |
| 보상·제어·성공·안전·무작위화 | 기존 계약 | 그대로 |

구형 체크포인트에 CLI만 붙여 Q/replay의 의미를 바꾸지 못한다. 초기 Q 갱신0·
네 optimizer 상태0·온라인/성공/n-step 은행0인 입력만 준비 도구가 받는다.
Q1/Q2/두 target의 첫 입력 행렬에 가중치0인 열을 추가하고 critic 정규화의
새 평균0·분산1을 넣는다. Actor·jaw·actor 정규화와 네 optimizer는 보존한다.
새 설정은 체크포인트·replay의 계약에 저장되며 재개 시 자동 복원된다.

## 사용법

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_critic_episode_clock_actor.py \
  --initial-checkpoint /absolute/path/to/pristine-inputs/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/pristine-inputs/training_manifest.json \
  --waypoints /absolute/path/to/pristine-inputs/waypoints.json \
  --output-dir /absolute/path/to/unique-new-inputs
```

이 준비 입력으로 기존 batched staged SAC 실행에
`--critic-episode-clock task-remaining`을 지정한다. Q 영상 exporter도 새 계약을
복원하지만 실제 기록된 `critic_episode_remaining` 없이는 표시를 거부한다.
평가 종료 시각에서 역산한 가짜 시간으로 Q를 계산하지 않는다.

## 현재 검증과 남은 확인

기존 관련65개 CPU 검사가 통과했고, 기록 schema 검사까지 확장한78개도
통과했다. GPU 통합 검사1개는 제외했다. 실제 전체
SAC trainer를 복원해 기존 TRAIN1,240상태에서 초기 actor4337의 목표·binary
jaw가 같고, 새 모델·네 optimizer와 빈 학습 은행이 정확히 복원됨을 확인했다.
같은 상태의 새 시간값0·0.5·1에서 Q1/Q2/두 target의 초기 함수도 보존됐다.
새 TRAIN1,536조건은15개 이전 계획과 seed가 겹치지 않고 원래 DEV128과
17개 wave를 유지한다. 독립 FINAL은 사용하지 않았다.
[실제 trainer 복원·새 계획](assets/rl_v2_task_remaining_fulltrainer_and_plan_20261007.json).

**실제 환경의 그룹 생성과 기록까지 최소 확인했다.** CPU 물리의 원래 DEV
4조건·1스텝 frozen 검사에서 실제 horizon900·정착 카운터·시간값112회,
행동 전/후 HDF4행의 시간 차이1/900과 계약·578차원 pilot을 확인했다.
Actor/Q/replay는0으로 유지됐다. 실제 검사에서는 새 그룹의 차원 검사 선언과
HDF recorder의 새 필드 선언 누락을 발견해 수정했다. 앞선 실패 writer는
종료 후 새 고유 검사에서 재확인했다. Isaac이 실패에도 exit0을 반환했으므로
`failure.json`과 `status.json`으로 성공 여부를 확인했다.
[실제 환경·카운터·HDF 정렬](assets/rl_v2_task_remaining_actual_group_runtime_20261007.json).
검사 writer와 관리 프로세스 종료 뒤 고유 폴더의 계약·로그3개만 기존 Drive로
업로드하고 파일별 크기·MD5를 확인했다. Raw HDF/관측 trace는 전송하지 않았다.
[종료 로그 백업 근거](assets/rl_v2_task_remaining_closed_logs_backup_20261007.json).

이 검사는 파지 단계 진입 전의 한 스텝으로 **실제 파지·시간 초과 자동 reset이나
학습 성능 검증은 아니다.** 현재 GPU3의 비교6개는 기존 설정으로 계속 진행 중이다.
완료한 전체 DEV 비교를 판단한 뒤 별도 고유 학습에 적용해야 한다.
현재 최고33/128은 성공 명령 유지 SAC의 결과이며 이 시간 입력의 결과가 아니다.

실행 시 [기존 Drive 관리자](RL_GOOGLE_DRIVE.md)의 checkpoint·계약·로그 범위,
300초 업로드·크기/MD5 검증·검증된 오래된 checkpoint만 정리·최근2개 보존을
사용한다. Raw replay/HDF는 로컬에 남기며 종료 로그는 writer 종료 뒤 검증한다.
