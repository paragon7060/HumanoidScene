# 관절 목표 변화 범위의 기구학 진단

보상과 탐색 잡음 외에 **정책이 허용하는 자세 변화 범위**도 점검했다.
종료된 actor1899의 상단 오른쪽 TRAIN 작업 위치 진단에서 baseline·바깥쪽3cm
후보 중 초기화가 유효했던6경로의 첫 안전 상태와 마지막 안전 상태, 총12상태를
사용했다. 원래 진단128요청 전체의 새로운 성공률을 계산한 것이 아니다.
DEV·FINAL 기록은 이 계산에 사용하지 않았다.

## 물리 기록과 모델의 대응

S63·Leju 두 손가락 URDF를 전체 root부터 shoulder·wrist·closed TCP까지 계산하고
관측의 실제 TCP와 비교했다. 최대 위치 차이는0.0000117m, 방향 차이는
0.00000080rad였다. 관측 자세를 기준으로 임의의 mount 변환을 맞춰 만든
동일성 검사가 아니라, 실제 관절20개와 모델의 전체 chain에서 독립 계산했다.

각 상태의 waist·torso·base는 측정값으로 유지했다. 팔별7관절에서 실제 URDF
관절 한계와 해당 정책의 absolute goal 범위를 함께 제한했다. 목표는 관측의
서로 다른 두 flap 중간지점과 nominal closed gripper 축의 flap 법선 정렬이다.
위치 오차와 축 오차를 최소화하는8개 시작점을 비교했다.

## 후반의 안전한6상태 비교

파지 단계 첫 상태는 아직 팔이 접근하지 않아 instant goal window로 중간지점에
도달할 필요가 없다. 이를 실제 파지 가능성 부족으로 해석하지 않는다.
아래는 별도로 분리한 **후반 안전 상태6개**의 정적 계산이다.

| 목표 범위 | 왼손 위치·축 충족 | 오른손 위치·축 충족 | 양손 모두 충족 |
| --- | ---: | ---: | ---: |
| 현재 nominal anchor의 normalized 반경0.15 | 0/6 | 0/6 | 0/6 |
| 같은 nominal anchor의 가상 반경0.30 | 0/6 | 6/6 | 0/6 |
| 학습된 actor689 body goal을 기준으로 한 가상 반경0.30 | 4/6 | 6/6 | 4/6 |

충족은 위치15mm 이내·nominal closing axis15° 이내라는 **진단용 수치**다.
실제 파지 조건을 바꾸지 않았다. Normalized 반경은rad 또는m가 아니며,
관절 목표의 기존 center·scale을 거쳐 각 관절의 실제 범위가 결정된다.
현재 반경을 유지한 계산에서도 전체 경로의 동적 도달이 불가능하다고 증명한 것은 아니다.
Anchor는 이후 상태에 따라 달라지고 waist·torso를 실제로 조절할 수도 있다.
다중 시작 IK의 실패 역시 전역적으로 해가 없다는 증명이 아니다.

후반 baseline 경로의 예에서는 왼손 위치 오차가6.58cm→0.14cm,
axis 오차24.44°→7.66°로 줄었다. 다른 바깥쪽 후보는 넓힌 계산에서도 왼손
5.54cm가 남아 전부 해결되지 않았다. 더 넓은 변화와 좋은 초기 동작을
함께 검토할 이유가 있으며, 단순히 잡음 크기만 올리는 것과 차이가 있다.

## 실제 학습에 연결할 때 유지할 조건

별도의 [학습된 body anchor·반경0.30 제어 옵션](RL_V2_REANCHORED_BODY_CONTROLLER_20261007.md)을
구현했다. Actor689의 실제 body goal을 frozen actor-only anchor로 사용하고 새
correction mean19개는0에서 시작한다. 실제 TRAIN 관측538개에서 초기 greedy 목표·
jaw·decode 명령 보존과 저장 후 복원을 확인했다. 새 학습의 물리 성공은 아직 검증 전이다.
기존 팔 탐색·출력 포화 완화 비교와 구분해 새 Q·replay·optimizer에서 비교한다.

몸체 목표 범위·controller가 달라지면 Q·target Q·critic 정규화·optimizer·replay·
성공 bank는 새로 시작한다. 이전 requested goal을 다른 의미의 action으로
재표시하지 않는다. 원래128개 전체 개발 평가와 별도의 새 TRAIN 조건으로
실제 양손 pad 접촉·hold·들기·랙 충돌을 다시 확인해야 한다.
박스·base·배경·firm dynamic flap 무작위화와 성공·안전 기준은 유지한다.

이 계산은 충돌, 모터 추종, flap 변형과 pad 접촉을 시뮬레이션하지 않았다.
IK 해나4/6을 물리 파지 성공이나 일반화 성공률로 세지 않는다.
해를 Q replay 또는 BC label에 넣지도 않았다. 실행 중인 학습과 모든 원본은 그대로다.

[12상태 전체 근거와6개 후반 상태 집계](assets/rl_v2_closed_TRAIN_local_goal_envelope_FK_20261007.json),
[최종 평가의 실제 실패·탐색 진단](RL_V2_PHYSICAL_EXPLORATION_DIAGNOSIS_20261007.md).
