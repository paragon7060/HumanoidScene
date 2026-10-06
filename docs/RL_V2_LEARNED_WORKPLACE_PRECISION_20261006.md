# 상단 오른쪽 진입 위치 진단과 정밀 파지 보상 비교

새 SAC의 첫 학습 후 전체 개발 평가는 **15/128 → 29/128**로 개선됐지만
상단 오른쪽은 **0/32**였다. GPU3 학습을 유지하면서, 같은 학습 정책의
작업 위치를 비교하고 남은 손의 파지 실패를 분석했다. 목표는 박스·base·주변
배치 무작위화를 유지한 네 구역의 실제 양손 파지다. 아직 달성하지 못했다.

## 실제 진입 위치 비교

Actor689/Q4802 정책으로 새로운 TRAIN 시작 조건 **16개**에 각각
작업 위치 **8개**를 적용했다. 네 구역 각4조건이며 물리 시도는 총128개다.
128개의 독립 배치나 DEV/FINAL 성공률로 해석하지 않는다. 후보끼리 요청한
시작 조건은 같지만 flap 성질과 접촉 이력까지 동일하게 통제한 비교는 아니다.

기존 작업 위치와 바깥쪽3/6/9cm, 좌우6cm·바깥쪽3cm·회전±6.88°를 비교했다.
전체128요청에 초기화 실패도 포함했다. 178개 모델·정규화 tensor와 업데이트
횟수는 모두 그대로였고 replay는0이었다. 종료된 데이터·로그는 기존 Drive
연결로 업로드하고 검증했다. 별도의 Q 학습 데이터로 가져오지 않았다.

| 구역 | 관측된 후보 결과 | 판단 |
| --- | --- | --- |
| 중간 왼쪽 | 기존 위치2/4, 바깥쪽3cm도2/4 | 작은 진단 표본에서 기존 위치 유지 가능 |
| 중간 오른쪽 | 기존 위치2/4, 초기 무효1건 포함 | 기존 위치 유지, 안정적 일반화 증거는 아님 |
| 상단 왼쪽 | 바깥쪽3cm1/4, 기존 위치0/4 | 별도 새 TRAIN 재확인이 필요한 후보 |
| 상단 오른쪽 | 모든 후보0/4. 바깥쪽9cm는4건 모두 안전한 시간 초과 | 위치 조정만으로 양손 파지는 해결되지 않음 |

후보별 모든 성공·충돌·시간 초과·초기화 실패는
[전체 진단 근거](assets/rl_v2_learned689_workplace_search_20261006.json)에 있다.
소수 사례에서 고른 위치를 전체 평가 성능으로 승격하지 않았으며, 현재
주 학습의 regional waypoint는 변경하지 않았다.

## 남은 손이 실패한 이유

상단 오른쪽32요청 중 초기화가 유효한26경로를 종료 후 분석했다.
정책 목표를 실제 명령으로 복원한 최대 차이는 **0.00002385**였다.
안전한 왼손 근접 상태 **2,739개에서 실제 왼손 닫기 명령은0회**였다.
이 상태들은 같은 경로에서 연속된 관측이며2,739번의 독립 시도가 아니다.
오른손 닫기 명령은 안전 근접 상태에서2,270회였다.

바깥쪽3cm의 한 사례는 오른손이 실제 파지했지만 왼손은 열려 있었다.
종료 시 rigid TCP 기준 표면 거리는 왼손3.41cm·오른손0.086cm,
nominal 닫힘 축 오차는 각각50.5°·5.6°였다. 다른 안전한 시간 초과 사례에서도
왼손 위치·정렬 오차가 남았다. 이 geometry는 실제 관측의 calibrated TCP와
panel 관계에서 계산했고, 움직이는 finger pad 자체의 좌표는 아니다.
실제 파지 여부는 기록된 filtered pad force·영역·opposed 조건으로 별도 확인했다.

따라서 왼손 닫기 탐색과 정확한 손 위치·방향 학습이 함께 필요하다.
단순히 충돌 기준을 완화하거나 닫기 자체에 보상을 주는 방법을 적용하지 않는다.

## 정밀 보상 SAC 비교

기존 선택형 정밀 capture를 사용한다. 손 접근0.22m와 랙 앞 진입 보상은
유지하고, 마지막 포획 위치의 거리 스케일만 **10cm → 2.5cm**, 손 합성은
평균에서 **0.25(sL+sR)+0.5·min(sL,sR)**로 바꾼다. Capture 가중치0.5와
실제 접촉·성공·충돌 기준은 같다. 한 손만 좋은 자세에 머무는 문제를 줄이고
마지막 수 cm에서 약한 손의 위치를 더 강조하려는 비교다.
개선된 성능이 확인됐다는 뜻은 아니다.

[초기화 도구](../scripts/rl/prepare_actual_flap_reward_actor.py)는 기존
actual-flap 학습 actor를 옮기되 보상이 달라지는 만큼 Q·target Q·entropy·
critic 정규화·네 optimizer·replay·성공 bank를 새로 만든다. 기존689회 actor
업데이트는 출처로 기록하며 새 학습 카운터는0이다. 같은 native audit 입력,
regional waypoint, 관측·행동 좌표, 물성·무작위화·안전 조건을 엄격히 확인한다.
실제 성공 TRAIN 관측538개에서 초기 실행 목표가 bit 단위로 같고 jaw 명령
불일치0, body 오차0을 확인했다. 관련29개 검사가 통과했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_actual_flap_reward_actor.py \
  --checkpoint /absolute/path/to/closed-actual-flap-checkpoint.pt \
  --training-manifest /absolute/path/to/matching-training-manifest.json \
  --waypoints /absolute/path/to/matching-regional-waypoints.json \
  --native-seed /absolute/path/to/closed-lower-audit.hdf5 \
  --native-seed /absolute/path/to/closed-upper-audit.hdf5 \
  --output-dir /absolute/path/to/unique-precision-initialization
```

보상 계산은 [정밀 capture 설정](RL_V2_PRECISION_CAPTURE_PROFILE_20261006.md),
대표 성능·영상은 [중간 보고](RL_V2_GRASP_INTERIM_SUMMARY_20261006.md)를 참고한다.
새 비교는 원래128개 DEV 요청과 별도의 새 TRAIN 배치를 사용하며,
독립 FINAL은 학습·선택에 사용하지 않는다. 현재 GPU3 주 학습은 계속 유지한다.
