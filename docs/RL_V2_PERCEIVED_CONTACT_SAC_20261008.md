# 양손 파지를 위한 관측 기반 접촉 탐색 SAC

2026-10-08 KST. 목표는 박스·base 시작 상태 무작위화를 유지한 **학습된
SAC 정책의 양손 flap 파지**다. 현재 일반화 성공은 확인하지 못했다.

## 왜 바꾸는가

기존 전체 평가에서는 학습 시간이 늘어도 성공을 잃는 경향이 있었다.
전체 팔 정책은 초기11/128에서 첫 학습 후10/128, servo guard는14→13→8,
uniform은14→8→4→1이었다. 기존 원본 실행도 TRAIN3,072개 조건을 마친
마지막 평가에서5/128이었다. 작은 박스 일부에 성공이 집중되고 중간 크기
박스는 성공하지 못했다. [마지막 원본 평가](assets/rl_v2_all6_eighth_learned_full_DEV_20261008.json).

실제 종료된 TRAIN117개 유효 경로를 보면 손이 가까워지는 것만으로 안전한
진입·닫기·유지·들기를 연속해서 만들지 못했다. 상단 왼쪽과 중간 크기
왼쪽에는 두 손이 가까웠는데도 실패한 경로가 있었다.
[실제 손 거리와 닫힘 분석](RL_V2_ALL6_TRAIN_FAILURE_MODES_20261008.md).

## 이번 변경

새 프로필 `full-arm-contact-exploration`은
`arm20-gentle-rest-greedy`에서 이미 뽑힌20% TRAIN 탐색 episode에만
관측 기반의 연속 접촉 시도를 추가한다. 나머지80%는 현재 정책의 greedy
수집을 유지한다. 새 난수 선택은 없다.

1. 두 손의 배정 flap 중점 거리가 모두22cm 안으로 들어오면 관측된 flap의
   접근 가능한 표면 지점을 정하고 랙 앞 진입 위치를 향한다.
2. 측정된 손 자세를 이용해 닫힘 축을 panel 법선에 정렬하며 표면에 접근한다.
3. 두 손이 목표18mm·축 오차0.25rad 조건에 들어오면 기존 jaw gate가
   허용하는 닫힘을 시도한다. 닫힘 명령이8 control tick 유지되면25mm 들기를
   **시도**한다. 닫힘 명령은 실제 접촉이나 파지 성공의 증거가 아니다.
4. 측정 관절과 pending PD target으로 매 tick 작은 관절 목표를 만들고,
   목표 lead를0.08rad로 제한한다.180 tick 또는 들기40 tick 후에는 원래
   정책 탐색으로 돌아온다.

입력은 기존 측정 관절·TCP·랙 좌표·알려진 박스 형상·flap perception이다.
접촉 센서·pinch·reward·success·critic 관측으로 탐색을 결정하지 않는다.
TRAIN에서 생성한 bounded21차원 URDF 목표가 실제24차원 명령으로 정확히
변환되므로 실제 환경 reward와 함께 SAC replay에 저장한다.

**평가와 actor/Q target 계산에는 이 접촉 탐색이 없다.** TRAIN 보조 동작으로
성공한 수와 학습된 greedy 정책의 평가 성공률은 따로 기록한다. 성공한
동작은 실제 TRAIN 경험으로 학습하는 것이며, 탐색 규칙을 평가에 붙여
정책 성공으로 표시하지 않는다.

## 유지한 설정

S63·Leju twofinger, 중력보상18관절, torso pitch 고정·XZ 이동, base 접근 후
hold, 팔 전체 URDF action 범위, 기존 actor518/critic578/action21을 유지한다.
기존 두 demo로 초기 actor만 복원하며 Q·replay·성공/return bank·optimizer는
새로 시작한다. 강한 성공 유지 항의 계수1은
[직전 비교 실험](RL_V2_URDF_STRONG_SUCCESS_SAC_20261008.md)과 같다.

박스 위치·base 시작·배경·단단하지만 움직이는 flap randomization, 기존 reward,
양쪽 opposing pad 접촉5N·0.25초 유지·8mm clearance 성공 기준을 유지한다.
랙10N·장애물5N·self-collision OFF도 같다. Box 고정, curriculum,
성공 기준 완화, DEV/FINAL 학습 데이터 유입은 없다.

학습은 원래 여섯 위치·크기 조합을 균형 있게 뽑는65 wave, TRAIN6,144개
조건·wave당128환경, replay200만이다. TRAIN384개 조건마다 같은 전체
DEV128을 보조 없는 greedy 정책으로 평가한다. 평가 무효 초기 상태도
128요청의 분모에 남긴다. 중간 크기 박스와 각 구역의 결과를 따로 확인하고
최종 후보는 아직 사용하지 않은 독립 FINAL에서 확인해야 한다.

## 구현과 확인

- [탐색기](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/perceived_contact_exploration.py),
  [SAC 연결](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/urdf_perceived_contact_sac.py).
- 관련 검사46개 통과. 실제 초기 모델 저장·TRAIN 복원·Q 영상용 복원 일치,
  기존 actor/normalizer29 tensor 정확히 동일, 모든 초기 모델75 tensor 유한,
  replay·Q·성공 bank0 확인.
  [초기화 근거](assets/rl_v2_URDF_perceived_contact_initial_verified_20261008.json).
- 종료된 실제 TRAIN117경로의 시작·최근접·마지막 관측351개에서 명령을
  읽기 전용으로 구성했다.161개에서 탐색 제안이 활성화됐고 FK 최대 위치
  오차는0.0162mm였다. 기존 jaw gate·비팔 좌표·bounded 명령·decoder 일치,
  model·관측·RNG 보존을 확인했다. Critic을NaN으로 넣어도 읽지 않는다.
  [관측 기반 검사](assets/rl_v2_perceived_contact_closed_TRAIN_rehearsal_20261008.json).

위 정적 검사는 실제 새 파지 성공이나 성능 개선의 증거가 아니다.
실제 물리 수집과 전체 평가를 계속 확인해야 한다.

## 실행·보관

고유한 frozen checkout와 실행 폴더를 사용하며 여유 GPU에
`CUDA_VISIBLE_DEVICES`와 Kit 단일 GPU를 함께 지정한다. 기존 활성 학습은
유지하고 다른 사용자 파일·프로세스를 변경하지 않는다. 실행 준비 문서만으로
학습이 시작됐다고 판단하지 않는다. 실제 PID·manifest·초기 모델을 대조한다.

기존 Drive 연결로300초마다 checkpoint·계약·로그를 업로드하고 크기/MD5
검증된 오래된 checkpoint만 정리하며 최근2개를 보호한다. Writer 종료 뒤
로그를 검증한다. Raw HDF/replay는 로컬에 남고 자동 삭제되지 않는다.
[보관 범위](RL_GOOGLE_DRIVE.md).
