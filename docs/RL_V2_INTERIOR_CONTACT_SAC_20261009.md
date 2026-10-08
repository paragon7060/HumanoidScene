# Flap 모서리를 피해 잡는 SAC 비교

목표는 박스와 base 시작 상태가 달라도 양손 flap 파지·유지·들기에 성공하는
학습된 SAC다. 최근 학습 후 전체평가 최고는13/128회이고 중형 성공은0회다.
기존 장기 학습과 몸통 유지 v5 비교를 계속하면서, 접촉 지점 문제를 v6로 분리한다.

## 실제로 확인한 실패

닫힘 추종을 강화한 v4의 첫 TRAIN128에서 탐색25경로는 실제 파지/성공0회였다.
닫기를 시도한8경로는14번 모두5–7틱 뒤 다시 열렸다. 거리12mm 조건이 원인이었고,
축 오차 초과는0회였다. 실제 그리퍼 닫힘은26.8–52.5%로 아직 덜 닫힌 상태였다.
상단11경로도 표면 접근 단계에 들어가지 못했다. 강화한 피드백만으로 해결되지 않았다.
[완료82,554행·실행 명령과 종료·닫힘 분석](assets/rl_v2_motion_feedback_completed_TRAIN_motion_analysis_20261009.json).

탐색8경로의 첫 닫기 TCP는 flap 옆 모서리에서 중앙값 좌0.57mm·우1.48mm 안쪽이었다.
성공10경로의 첫 실제 양손 파지는 좌18.10mm·우50.53mm 안쪽이었다. 실패에서도
손은 랙 안쪽을 향했다. 이 수치는 nominal TCP의 기하이고 실제 pad 위치 측정은
아니다. 서로 다른 경로와 단계이므로 모서리가 실패를 일으켰다고 확정하지 않는다.
[24경로의 실제 지점·방향·분석 한계](assets/rl_v2_motion_feedback_completed_TRAIN_panel_margins_20261009.json).

## 바꾸는 부분

![안쪽 접촉 지점과 완료 TRAIN의 실제 모서리 여유](assets/rl_v2_interior_contact_method_20261009.png)

- **목표 지점:** flap 접선 방향의 최소 여유를5→20mm로 늘린다. 손에 가까운
  패널 지점을 선택하고 한 시도 동안 패널 좌표에 고정한다. 중심점으로 강제하지 않는다.
  지점6mm 허용 오차만 고려하면 모서리까지 약14mm 기하 여유를 남기지만 실제
  pad 접촉이나 충돌 없는 파지를 보장하지 않는다.
- **몸통과 팔:** v5와 같은 몸통 X/Z 범위·양팔 SAC 분포와 진입 후 몸통 유지다.
  박스/base/flap 무작위화, 보상, 관절 속도·힘 제한, 양손 파지·유지·들기·안전 조건은 유지한다.
- **학습과 평가:** 기존20% TRAIN 탐색 선택에만 새 지점을 쓴다. 접촉 힘·파지·
  보상·성공 판정은 탐색 입력으로 쓰지 않는다. Actor/Q sampling·밀도·entropy와
  보조 없는 greedy 평가는 v5와 같다. 새 Q/replay/optimizer로 시작한다.

선택 옵션은 `scripts/rl/prepare_urdf_regional_goal_sac.py`의
`--servo-retention-profile full-arm-interior-contact`와
`--body-behavior arm20-gentle-rest-greedy`다. 기존 옵션은5mm 지점과 원래 동작을 유지한다.
Artifact는 `staged_actual_flap_URDF_upright_interior_contact_exploration_sac_v6`이며
모델 재개·평가 영상 복원·완료 replay 진단에도 별도 계약을 연결했다.

## 검증과 실제 실행 구분

관련 검사79개와 실제 초기 checkpoint 저장·학습 재개·영상용 모델 복원이 통과했다.
기존 v5와actor/정규화29개·몸통 범위 buffer·frozen source가 같았다. 검증된 완료
TRAIN의384상태에서 초기 greedy 목표21개가 모두 정확히 같았고 density와
entropy도 유한했다. 분석 상태나 명령은 학습에 넣지 않았다. 이는 초기화·범위
검증이며 새 물리 성공이나 학습 개선은 아직 확인하지 않았다.
[실제 저장·재개·초기 정책 대조](assets/rl_v2_URDF_interior_contact_initial_verified_20261009.json).

완료된 실제 v4 TRAIN의 탐색 시작25상태도 대조했다. 새 양손 목표50개 모두
알려진 패널 접선 경계에서20mm 안쪽이었고 기존 목표 대비 이동은최대2.12cm였다.
이는 후보 지점의 기하 검증이며 v6가 실제로 실행한 궤적이나 파지 성공이 아니다.
[실제 탐색 시작 상태의 후보 지점](assets/rl_v2_interior_contact_actual_handoff_geometry_20261009.json).

원래 첫 다섯 배치의 TRAIN384·초기/학습 후 전체DEV각128·replay25만으로 비교한다.
중간/상단·좌/우와small/medium의 원래 여섯 조합을 유지하고 독립FINAL은 아직
사용하지 않는다. 기존 일곱 장기 비교는TRAIN6,144·replay200만으로 계속한다.
GPU3의 앞선 v4 writer/supervisor가 정상 종료하고 최종Drive 검증을 마친 뒤,
GPU 메모리·디스크를 다시 확인해 별도 v6를 시작하도록 CPU 대기 작업을 등록했다.
06:26 KST에 실제 PID589,524·빈 `CUDA_VISIBLE_DEVICES`·v4 writer3,191,064와
supervisor3,190,664의 생존을 확인했다. 고정 소스749개와 초기 checkpoint를
검증했으며 현재 새 v6는 대기 중이다. 기존 실행에 신호를 보내지 않는다.
시작 후 실제 런타임·첫 TRAIN·전체 초기/학습 후 평가·같은 모델 보존용 CPU
관찰 다섯 개를 자동 등록한다. 준비 파일의 존재를 실제 학습 시작으로 판단하지 않는다.
[실제 대기 PID·고정 소스·시작 조건](assets/rl_v2_URDF_interior_contact_actual_GPU3_queue_20261009.json).

기존 Drive 연결·300초 업로드·크기/MD5 검증·최신2개 checkpoint 보호를 재사용한다.
RawHDF/replay/영상은 checkpoint/log 전용 백업 범위에 없으므로 로컬에 보관한다.
종료 로그는 writer가 멈춘 뒤 검증하며 다른 사용자 파일·프로세스는 변경하지 않는다.

[최신 전체평가·실제 실행 상태·성공/실패 영상](RL_V2_SAC_PROGRESS_20261009.md).

Notion 중간보고에도 새 지점 선택 방법과 실제 대기를 기록하고 그림을 native
image로 올렸다. 기존 미디어77개와표5개가 그대로 남았고 새 그림을 포함한
78개 미디어·native table5개를 다시 확인했다.
[보고·그림·표 보존 검증](assets/rl_v2_interior_contact_Notion_verified_20261009.json).

07:04 KST 추가 확인: 같은 몸통 제어기를 쓰는 v5의 첫 TRAIN에서 IK 경계
오류가 발생해 v6의 우리 CPU 대기를 GPU 시작 전에 중단했다. v6의 실제 학습
시작이나 성공을 주장하지 않는다. 마지막 유효 목표를 유지하는 공통 복구
수정과81검사·실제 관측 상태 재계산이 통과했으며 수정 소스를 고정한 뒤
다시 연결한다. 기존 GPU 학습과 다른 사용자 실행은 유지했다.
[오류·수정·실제 상태 검증](RL_V2_UPRIGHT_CONTACT_SAC_20261009.md#0704-몸통-ik-경계-오류와-복구-수정).

07:12 KST에 수정 source3740689·749개Python 파일을 쓰는 새CPU 대기
PID1,283,615·빈GPU mask를 확인했다. 초기 actor/SAC분포·20mm지점·TRAIN384
설정은그대로이며v4의 정상종료·최종Drive검증 뒤 실제GPU3 여유를확인해
이어실행한다. 옛PID589,524 대기는중단됐고 새v6의물리학습은아직시작전이다.
[수정 실행과 새대기 PID](assets/rl_v2_URDF_upright_repaired_long_actual_launch_20261009.json).

07:34 KST에 앞선 v4가 정상 종료·최종 Drive 검증을 마친 뒤 새 v6가 실제로 시작됐다.
별도 폴더의 writer1,624,488·supervisor1,624,392·CUDA_VISIBLE_DEVICES=3·
수정 소스3740689/749개 Python을 확인했다. 원래 첫 다섯 배치·TRAIN384·전체 DEV각128·
replay25만·20mm 지점을 유지한다. 아직 초기화 중이며 물리 파지나 학습 개선은 미확인이다.
CPU 대기 coordinator가 exit0으로 끝난 것은 시작 완료이며 새 GPU 학습 종료가 아니다.

새 CPU 관찰 다섯 개도 등록했다. 첫 학습 후 평가 관찰은 초기 progress 파일 생성 전에
읽어 종료하는 문제를 고쳐 PID1,652,910으로 교체했다. 수정 장기 v5의 같은 CPU
관찰도 PID1,651,919로 복구했다. GPU writer나 Drive 업로드는 중단하지 않았다.
[실제 시작·고정 소스·대기 완료·관찰 PID](assets/rl_v2_URDF_interior_contact_repaired_actual_launch_20261009.json).

07:40 KST에 초기 전체평가 step1·Actor/Q0·온라인0에 진입했다. 실제 agent/progress/
manifest·원래 다섯 배치와749개 고정 소스가 일치했고 중력보상18관절을 확인했다.
VRAM2,791MiB와 첫 Drive 계약 검증도 확인했다. 이는 실행·초기 평가 진입이며
파지 성공이나 학습 개선은 아니다. 학습 후 전체평가와 같은 모델의 실제 파지로 판단한다.
[실제 첫 평가·설정·Drive 검증](assets/rl_v2_URDF_interior_contact_repaired_SAC_actual_startup_20261009.json).

Notion 보고에 이번 몸통 추종 그림을 native image로 추가했다. 기존 미디어78개와
표5개를 그대로 보존했고 현재 미디어79개·native table5개를 다시 확인했다.
[보고·그림·표 보존 검증](assets/rl_v2_upright_tracking_and_contact_launch_Notion_verified_20261009.json).

## 수정 실행의 전체 초기 평가 완료

08:20 KST에 수정 v6의 초기 전체128조건이 완료됐다. 성공14·안전 위반44·
시간 초과57·초기 무효13회이며, 중간 왼쪽small9·상단 오른쪽small5·다른 네 조합
0회다. 실제 저장 모델/정규화가 초기 입력과 같고 actor/Q0이다. 이는 학습 효과가
아니며 TRAIN20% 탐색의 안쪽20mm 지점은 이 greedy 초기 평가에 사용되지 않는다.
첫 TRAIN을 실제 시작했고 완료 수집과 학습 후 전체 결과를 기다린다.
[전체 초기 평가·같은 실제 모델](assets/rl_v2_URDF_interior_contact_repaired_first_full_DEV_20261009.json).
