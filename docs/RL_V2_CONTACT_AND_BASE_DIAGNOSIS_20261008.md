# 가까운 손이 실제 양손 파지로 이어지지 않는 원인 진단

현재 전체 개발 평가의 성공 흐름은35→19→23/128이다. 직전보다 회복했지만
학습 전보다 낮다. 아래 접촉 진단은 이 live 학습 데이터가 아니라 **정상 종료한
고정 정책의 크기별 TRAIN 접근 진단**을 읽은 결과다. 평가 경험이나 합성
성공을 학습에 넣지 않는다.

## 랙 충돌은 파지 단계에서 발생했다

수정 후 동일 TRAIN16조건×접근 후보8개의128요청 중 초기 무효10개를 제외한
닫힌 HDF118개를 읽었다. 모델·normalizer198개와 actor·Q·replay0 고정을
확인했고, 각 HDF의 reset 전 마지막 pinching·stable·success·unsafe가 실제
terminal 기록과 모두 일치했다. 원본 HDF의 inode·크기·mtime도 변하지 않았다.

안전 위반77개는 모두 robot–rack 충돌이었고 접근을 끝낸 held-grasp 단계였다.
종료 시 최대 힘을 기록한 링크는 오른쪽 그리퍼 몸체45개, 왼팔 link4 9개,
왼쪽 그리퍼 몸체4개, 오른팔 link7 13개, 오른팔 link4 6개다. **이 링크는 실패
시점의 최대 힘이며 최초 접촉이나 충돌을 일으킨 첫 관절을 뜻하지 않는다.**
특히 상단 오른쪽의26개 충돌 중24개는 오른쪽 그리퍼 몸체였다. 팔과 그리퍼의
진입 위치를 함께 확인해야 하며 안전 threshold를 완화해 해결하지 않는다.

## 근접·닫힘·포획 점수와 실제 접촉은 다르다

상단 왼쪽의 원래 접근 후보 env16은 손–flap 표면 거리 약5.93mm/0mm,
접근 잠재값0.980·정렬0.966·포획0.894였다. 마지막 동작은 양손 닫힘이었고
실제 그리퍼 닫힘 비율도1.000/0.999였다. 그러나 배정된 flap의 pad 힘은 다음과
같았다.

| 손 | pad 1 | pad 2 | 마지막 실제 파지 |
| --- | ---: | ---: | --- |
| 왼손 | 0N | 1.63N | 성립하지 않음 |
| 오른손 | 34.10N | 35.81N | 성립 |

양손 opposing flap 파지는 중간에3개 관측에서만 성립했고 연속 최대는
0.033초였다. 요구한0.25초 유지와8mm clearance는 충족하지 못했고 시간
초과였다. **가까워진 점수나 닫힘 명령은 성공이 아니다.** 현재 보상은 이미
약한 pad·약한 손의 실제 접촉 진행과 파지 이벤트를 반영하며, 닫힘 명령
자체에는 보상이 없다. 이 사례만으로 해당 보상 계산이 틀렸다고 단정하지
않는다. 손가락 위치·지속 접촉·진입 안전성과 Q의 행동 선택을 함께 진단한다.

전체 유효118회 중109회는 opposing 양손 파지가 한 번도 성립하지 않았다.
그중 medium 유효24회는 모두 한 번도 성립하지 않았다. 9회에서 잠깐이라도
성립한 것과 실제 성공4회를 구분한다. 원래 요청128개 분모의 실패를 삭제하지
않았으며 포획 잠재값 하나를 성공으로 재라벨링하지 않았다.

## 상단 동작의 base 안정성은 추가 원인 후보

관측에 기록된 held 단계의 base 각속도 norm 중앙값은 중간 구역 각 그룹에서
약0.006rad/s, 상단 왼쪽0.733·오른쪽0.679rad/s였다. 같은 관측의 랙 상대
orientation으로 계산한 handoff 대비 최대 기울기의 그룹 중앙값은 상단
왼쪽3.70°·오른쪽3.57°였다. 이는 **30Hz로 표본화한 상태의 요약**이다.

일정 방향으로 큰 각속도가 기록되더라도 표본 자세는 수 도 이내이므로,
물리 substep의 진동·표본화 효과인지 제어기 문제인지 확인하기 전에는
중력보상 실패나 torso 붕괴라고 확정하지 않는다. 각속도를 단순 적분해
보이지 않은 움직임을 추정하지 않는다. 실제 하위 스텝의 자세·속도·wrench와
그리퍼 상대 오차를 함께 측정하는 것이 다음 진단이다. 이번 작업에서는
센서·물리·제어기·정책·보상을 변경하지 않았다.

## 매 물리 스텝의 원래 제어를 관찰하기

`--base-substep-trace-env-indices`는 원래 floating-base drive가 계산한 명령을
그대로 실행한 뒤, 다음 물리 스텝 전에 root 자세·세계 속도·실제 PhysX view의
자세/속도·명령 force/torque·target·관성 추정값을 기록한다. Quaternion은wxyz이고
세계 힘/torque와 body 힘/torque를 각각 표시한다. **명령 torque는 측정된 반력이나
PhysX가 실제 달성한 torque가 아니다.** 캐시와 raw view의 차이, 물리 스텝 간
자세 변화와 속도, 제어 입력의 부호를 확인하기 위한 read-only 도구다.

실행은 완전한 고정 TRAIN 접근 진단128요청에서 선택한 최대8개 환경만 허용한다.
학습 모드·축소된 환경·일반 DEV/FINAL 경로에서는 거부한다. 원래 양손 파지·
유지·clearance·안전 기준과 무작위 배치는 그대로 유지한다. 실제 제어기의
`apply`를 같은 입력으로 정확히 한 번 호출하고 반환값을 보존하며, 종료 시 원래
메서드를 복구한다. 로봇 상태를 쓰거나 접촉 센서를 추가하지 않는다. 유효하지
않은 상태는 로그에 표시하고 물리 상태를 유효한 값으로 교체하지 않는다.

원래 frozen TRAIN 실행 명령에 다음 옵션을 추가한다. 대표 영상 선택은 선택형이다.

```bash
--base-substep-trace-env-indices 0 16 24 32 40 64 \
--eval-video-env-indices 0 16 24 32 40 64
```

실행 폴더의`base_substep_trace.log`는 JSONL이다. 원래 접근 후 제어 스텝의
각 physics apply 호출을 기록하며 원래 배치가 무효이거나 종료한 환경의 기록은
추가하지 않는다. 마지막`closed`행을 확인한 뒤 분석한다. 기존 최소 백업은
writer 종료 후 이 로그도 업로드·체크섬 검증 대상으로 포함한다. 영상과 raw
HDF/replay는 이 checkpoint/log 백업 옵션으로 전송하지 않으므로 별도로 보존한다.

관련36개 검사에서 명령·반환값·로봇 상태 보존, raw/cached 상태 구분,
world/body torque 표현, inactive 환경 제외, 비유한 값의 원본 보존, 일반 학습·
축소 scope 거부와 종료 로그 백업을 확인했다. **이는 도구 검사이며 base 진동의
원인이나 파지 성공 개선을 증명하지 않는다.** 실제 고정 정책의 원래128요청을
다시 실행해 물리 스텝별 측정과 전체 종료 결과를 함께 확인한다.

## 도구와 근거

`summarize_closed_contact_attempts.py`는 우리 소유의 정상 종료한 고정 정책
TRAIN 접근 진단만 읽는다. Live HDF/replay나 다른 사용자의 파일은 읽지 않으며
명령 형식464D actor·66D privileged·24D 실행 command와 모든 실제 terminal을
검사한다. 출력에는 각 시도의 pad 힘·접촉 영역·파지 지속·rack 최대 힘 링크·
표본 base 운동을 남긴다. 힘은 기존 critic feature의200N 상한을 명시한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/summarize_closed_contact_attempts.py \
  --experiment-dir /absolute/path/owned_normally_closed_frozen_TRAIN_probe \
  --output-json /absolute/path/new_contact_report.json
```

[원래118개 실제 접촉·terminal 일치·샘플 상태](assets/rl_v2_corrected_closed_contact_attempts_20261008.json).
[새 시작 조건의 크기별 재확인](RL_V2_SUPPORTED_SIZE_WORKPLACE_20261008.md).
[같은 전체 DEV의 실제 학습 추이](RL_V2_REGIONAL_INITIAL_DEV35_20261008.md).

박스·base·배경·움직이는 flap의 무작위화, rack10N·장애물5N·self-off와 원래
양손 접촉·유지·lift 기준을 유지했다. 모든 여섯 구역·크기의 안정적인 성공과
손대지 않은 독립FINAL은 미확인이며 목표는 완료되지 않았다.
