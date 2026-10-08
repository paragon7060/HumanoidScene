# 박스 밀림을 줄이기 위한 실제 flap 추종 진단

목표는 기존 박스·base·배경·움직이는 firm flap 무작위화를 유지한 SAC 양손
파지다. GPU3의 여섯 조합 학습은 계속한다. 아래 보정은 학습된 SAC와 구분한
별도 고정 정책 진단이다. 첫 중간점 추종 진단은 정상 종료했고 성공0/128이었다.
실제 접촉 영역에 맞춘 후속 방법은 아래에 기록한다.

## 전체 종료: 중간점 추종 진단은 성공0/128

12:13 KST에 원래 writer1476893의 정상 exit0·run complete를 확인했다.
원래 전체128요청은 성공0·안전 위반38·시간 초과76·초기 무효14회다.
안전 위반은 로봇–랙 충돌30·낙하8회였다. 84환경에서 보정30,017행을
실행했지만28,600행은 기존 affine 목표 범위에 걸렸고 닫기·들기 전환은0이었다.
모델·normalizer198개 및원래447개 Python SHA256은 **종료 당시** 그대로였다.
같은 원래 요청의 공통 유효114조건은 기존3회·진단0회 성공이다. Flap 추첨·접촉
이력까지 동일한 반복은 아니며 SAC 성능으로 집계하거나 이 방법을 채택하지 않는다.
[전체128·모델·소스·물리 조건 검증](assets/rl_v2_cartesian_flap_full128_closed_20261008.json).

정상 종료한 HDF의 유효114경로에서 전체24차원 명령을 모두 재구성했다.
최대 차이는0.00011921이었고 decoder·model·checkpoint·HDF SHA256은 보존됐다.
재구성한 전체 카운터도 실제 report와 일치했다. 삽입 단계에 들어간 환경은4개였고
실제 opposing 양손 pinch는 모든 경로에서0이었다. 클램프 횟수만으로 모든 목표가
도달 불가능하다고 단정하지 않는다.
[실제 모든 명령·단계·접촉 재구성](assets/rl_v2_closed_cartesian_flap_actions_audit_20261008.json).

| 종료 진단 영상 | 실제 결과 |
| --- | --- |
| [중형 왼쪽 env32](assets/rl_v2_cartesian_probe_20261008/env032_h264.mp4) | 왼손 gripper body–rack12.35N, 양손 pinch 없음 |
| [중형 오른쪽 env104](assets/rl_v2_cartesian_probe_20261008/env104_h264.mp4) | 오른손 gripper body–rack130.90N, 양손 pinch 없음 |

두 영상은 기존 물리 기록을 H264/avc1·yuv420p·faststart로 인코딩하고 전체 decode와
원본 보존을 확인했다. 선택 영상은 별도 점수가 아니며 학습된 SAC나 Q-value 영상으로
표시하지 않는다. [영상 형식·checksum·실제 종료 상태](assets/rl_v2_cartesian_probe_closed_video_evidence_20261008.json).

## 실제 성공 위치와 새 닫기 조건의 불일치

기존 정상 종료 TRAIN 진단의 성공3개를 확인했다. 실제 bilateral pinch85개 관측 중
양손 모두 중간점1.8cm·닫힘 축0.25rad 조건을 만족한 관측은0이었다.

| 기존 실제 성공 | 첫 양손 접촉 시 중간점 거리 좌/우 | 닫힘 축 오차 좌/우 |
| --- | --- | --- |
| 중간 왼쪽small env0 | 6.72cm / 4.50cm | 7.2° / 0.9° |
| 중간 왼쪽small env64 | 6.98cm / 4.38cm | 8.3° / 1.3° |
| 상단 오른쪽small env88 | 2.29cm / 4.81cm | 18.8° / 21.3° |

이 관측은 이미 접촉한 과거 상태다. 닫기 전의 반사실 물리 결과나 중형의 성공을
증명하지 않는다. 그러나 유효한 flap 접촉 영역을 중간점 주변의 작은 구로 제한하면
이미 확인한 실제 파지 위치도 제외한다. 이전 로그의`flap_distances`는 중간점이
아니라 **배정flap의 가장 가까운 표면까지 거리**임도 원래 계산 코드에서 확인했다.
수치 자체는 보존하고 실패 분포 보고서의 거리 이름을 바로잡았다.
[실제 접촉·중간점 좌표·critic terminal schema 대조](assets/rl_v2_existing_success_contact_geometry_20261008.json).

## 접촉 영역을 사용하는 후속 진단

기존 v1과 원래 학습의 관측·보상은 유지한다. 별도
`scripts/rl/frozen_cartesian_region_probe.py`는 같은 전체128 guard를 사용한다.
중간점38D 관측으로 두 손이22cm 이내에 들어오면, 해당 손의 위치를 알려진
flap 직사각형 영역 안에 투영해 접촉점 하나를 선택한다. Normal 좌표는 두 pad
사이의 panel plane에 두고 tangent 가장자리에서5mm 안쪽으로 제한한다. 한번
선택한 panel 좌표를 유지하면서 움직이는flap을 따른다. 관측 자체를 이동하는
최근접 점으로 바꾸지 않는다.

기존 neural 그리퍼 닫기와 production12cm gate를 유지한다. 추가 보정의 엄격한
위치·축 조건에 못 들어왔다는 이유로 원래 허용된 닫기를 다시 열지 않는다.
실제 pinch를 확인한 손은 측정한 위치·방향을 유지하며 flap을 추가로 회전시키지
않는다. 실제 opposing8tick 뒤2.5cm proof lift를 제안하는 privileged 교사 진단임은
그대로이며 standalone SAC·새 Q 행으로 집계하지 않는다.

팔 step0.02rad·기존 affine0.30 범위·비팔 neural 명령·무작위화·성공·안전 기준은
유지한다. 이 tag가 있는 결과는 원래 waypoint/Q 준비 경로에서 계속 거부된다.
관련21개 검사와 정상 종료한 기존 성공3경로의1,408개 입력에서 실제 decoder·
비팔 명령·목표 범위·원래 닫기 보존을 확인했다. 후자는 실행하지 않은 제안이며
새 접촉·들기 성공으로 표시하지 않는다.
[원래 실제 입력에서의 제안 검사](assets/rl_v2_cartesian_region_closed_success_rehearsal_20261008.json).

### 후속 진단 v2 실제 시작

13:01 KST에 GPU0의 고유 실행을 시작했다. 13:05에 실제 writer2332935의
소유자·고유 경로·CUDA_VISIBLE_DEVICES=0과 원래448개 Python SHA256,
새 진단 tag·원래전체128요청·접근 후보8개·물리/보상/DR/안전 계약 일치를
확인했다. 폴더는`GPU0_actual_flap_region_contact_frozen_TRAIN128_20261008_130058/`
`batch_sac_20261008_130059_2149c5`다. 당시 초기화 중이고 실제 control report와
새 접촉 결과는 아직 없었다. 시작 확인을 보정 실행 또는 성공으로 집계하지 않는다.
CPU 전용 확인기2356472는 같은 writer의 정상 종료 후에만 모델198개·소스448개·
원래 전체128결과를 판정한다. GPU3 여섯 조합 writer3987840은 계속 학습하고
small Q 보강 비교는23/128로 정상 종료했다. 원래 프로세스를 종료하지 않았다.
[실제 시작·manifest·확인 범위](assets/rl_v2_cartesian_region_actual_startup_20261008.json).

Notion 보고서에는 이전 진단의 재생 가능한 영상2개와 실제 성공 영역을 토대로
바꾼 방법, 전체 학습 평가11→10→7회와 small 비교23회를 구분해 기록했다.
기존media58개 내용·순서와 native표5개를 보존하고 새 native영상2개를 확인했다.
[Notion·기존 자료·새 영상 확인](assets/rl_v2_cartesian_region_and_latest_DEVs_Notion_verification_20261008.json).

## 확인한 원인

정상 종료한 이전 진단의 두 중형 사례에서 손–flap 거리 증가를 분해했다.
손에서 독립적으로 복원한 flap 좌표가 일치했고 배정 변경은 없었다.

| 사례 | 구간 | 박스 이동 크기 | 박스 회전의 flap 이동 기여 | 박스 기준 flap 이동 기여 |
| --- | ---: | ---: | --- | --- |
| 중형 왼쪽 env32 | 0.300초 | 4.97cm | 좌0.65cm / 우1.39cm | 좌0.44cm / 우약0cm |
| 중형 왼쪽 env96 | 0.667초 | 9.47cm | 좌6.62cm / 우3.06cm | 좌0.46cm / 우0.46cm |

좌/우는 각각 왼손·오른손에 배정된 flap이다. 각 기여는 벡터 합으로 전체 flap 이동을 재구성한다. 표의 크기는 방향이 달라
단순 합산하지 않는다. 이 두 사례에서는 flap 자체의 큰 변형보다 박스의 이동·
회전이 컸다. 어느 손의 접촉이 처음 박스를 밀었는지나 모든 실패의 원인까지
증명하지 않는다. [원본 유지·좌표·벡터 분해](assets/rl_v2_closed_medium_box_flap_motion_decomposition_20261008.json).

S63/leju-twofinger의 실제 관절각으로 계산한 calibrated closed TCP도 종료된
55시도의 6,134개 손 자세와 비교했다. 최대 위치 오차0.0161mm·회전 오차
0.00000132rad였다. 실시간 simulator Jacobian이나 새 보정 동작의 성공을
증명한 검사는 아니다. [실제 관절·TCP 좌표 대조](assets/rl_v2_closed_S63_measured_TCP_fk_audit_20261008.json).

![잡기 전 박스 이동과 박스 기준 flap 움직임](assets/rl_v2_box_vs_flap_motion_20261008.png)

## 첫 진단 v1의 설계

기존 neural 접근을 유지하다 두 손이 실제 배정 flap 중간점22cm 이내에 들어오면
팔만 보정한다. 랙 앞 진입 위치에서 닫힘 축을 panel 법선에 맞춘 뒤 실제 중간점을
추종한다. 팔의 제안 step은 최대0.02rad이고 기존0.30 affine body goal 범위와
기존 physical decoder를 유지한다. Base·몸통·머리·waist yaw는 원래 neural 명령이다.
기존 production12cm gate를 유지하며 실제 위치·축 정렬 후 닫기를 요청한다.

실제 opposing 양손 pinch를8tick 확인하면 측정한 잡은 위치에서 랙Z로2.5cm
들기를 제안한다. **이 진단의 유지·들기 판단에는 privileged 실제 pinch를 사용한다.**
따라서 성공하더라도 standalone SAC 또는 독립 일반화 성공으로 보고하지 않는다.
접촉 없이 닫힘 비율만 커진 경우나 같은 flap을 두 손으로 잡은 경우는 들기를
시작하지 않는다. 접촉을 잃으면 다시 파지 단계로 돌아간다.

원래 전체 TRAIN128 요청·접근 후보8개·기존 checkpoint를 사용한다. 중간 좌우
small/medium 각16개·상단 좌우small 각32개를 포함한다. 박스 고정·범위 축소·
성공 reset·controller gain 변경·safety 완화는 없다. 기존 실제 양손5N·0.25초 유지·
8mm clearance·랙10N·장애물5N·낙하10cm 및 self OFF를 그대로 판단한다.
Actor/Q/replay는0이고 모델·normalizer198개 고정 검사를 재사용한다.

`scripts/rl/frozen_cartesian_flap_probe.py`는 원래 전체 진단 guard와 runner를
재사용하고 process-local hook을 복구한다. Manifest·실제 report에
`frozen_cartesian_flap_probe`를 남기며, 기존 waypoint/Q 준비 경로는 이 변경된
결과를 거부한다. 진단의 실제 물리 명령과 저장 goal은 같은 decoder에서 생성한다.
오프라인 제안이나 평가·진단의 결과를 새로운 Q 경험으로 바꾸지 않는다.

기존 Google Drive wrapper·300초 검사·최근2개 보호를 재사용한다. 인증 오류가
계속되면 미검증 checkpoint를 유지한다. 다른 사용자의 파일·프로세스와 기존
학습 소스 worktree는 건드리지 않는다. 아래 실제 시작 기록에 고유 폴더와 PID를 기록했다.

## 첫 진단 v1의 실행 전 확인

관련17개 검사에서 TCP Jacobian의 위치·각도 변화, 비팔 명령 보존·정확한 goal
디코딩, 빈 닫힘·같은 flap의 들기 차단과 실제 opposing8tick 확인, 학습 거부·
hook 복구·기존 waypoint/Q 준비 거부를 확인했다. 추가로 정상 종료한 네 사례의
실제 입력411행에서 보정 제안을 계산했다. 모델·HDF는 그대로였고 원래 decoder와
비팔 명령이 일치했다. 제안을 물리적으로 실행한 데이터나 학습 성공으로 표시하지
않는다. [실제 입력의 오프라인 제안 확인](assets/rl_v2_cartesian_probe_closed_state_rehearsal_20261008.json).

## 첫 진단 v1의 시작 당시 기록

10/08 11:34 KST에 별도GPU0 실행을 시작했다. 실제 writer1476893의 소유자·고유
경로·CUDA_VISIBLE_DEVICES=0을 확인했다. 폴더는
`GPU0_actual_flap_cartesian_contact_frozen_TRAIN128_20261008_113429/`
`batch_sac_20261008_113431_535efb`다. Manifest의 진단 tag·원래128요청 일치와
원래447개 Python source SHA256을 확인했다. 당시 전체 물리 결과·종료 후198개
모델 고정 검증은 대기 중이었다. 이후 정상 종료 확인은 이 문서 맨 위에 기록했다.
GPU3의 여섯 조합 SAC와 Q 보강 SAC는 유지했다.
[실제 PID·장치·원래 요청·검증 범위](assets/rl_v2_cartesian_probe_actual_startup_20261008.json).

11:46 KST에는 같은 writer가 실제 control step121·13,794행까지 진행했고
유효한114환경 모두 base 정착 이후 단계에 들어갔다. Actor/Q/온라인 replay는
모두0이다. 측정 관절로 계산한 TCP의 실시간 최대 위치 차이는0.0163mm다.
이 시점에는 두 손22cm 진입 조건을 만족한 환경과 보정 명령 행이 아직0이므로
물리 rollout 시작과 보정 성공을 구분했다. 당시 전체128 종료 결과는 아직 없었다.
