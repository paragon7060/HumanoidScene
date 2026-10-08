# 중형 진입과 양손 닫기 선택의 분리 진단

목표는 박스·base·배경·움직이는 firm flap의 무작위화를 유지한 SAC 양손 파지다.
GPU3의 여섯 조합 학습은 유지한다. 아래는 닫기 선택을 분리하는 고정 정책
진단이며, 이 결과를 SAC 개선이나 독립 일반화 점수로 집계하지 않는다.

## 실제 실행 상태

2026-10-08 09:47 KST에 별도GPU0 실행을 시작했고 10:03 KST에 실제
writer437496의 소유자·고유 실행 경로·`CUDA_VISIBLE_DEVICES=0`을 확인했다.
고유 폴더는`GPU0_bilateral_near_close_frozen_TRAIN128_20261008_094705`다.
Manifest와 실제 learner report에 진단 tag가 있고 원래233개 소스 SHA256은
유지됐다. 전체128요청 진행 중이며 actor/Q/온라인/replay는0이다.
현재 물리 성공 결과와 종료 후198개 tensor 검증은 **대기 중**이다.

기존 Drive wrapper·300초 검사·최근2개 보존을 재사용한다. 백업 범위는
checkpoint·계약 metadata·종료 로그이며 raw HDF/replay/영상은 로컬에 남는다.
인증 실패로 검증되지 않은 원본을 삭제하거나 새 인증을 만들지 않는다.
기존 실행과 다른 사용자의 프로세스·파일을 건드리지 않았다.

## 정상 종료한 실제 기록에서 확인한 점

원래 중간 좌우 small/medium·상단 좌우 small의 TRAIN16조건×접근 후보8개,
총128요청을 완료했다. 초기 무효13회를 분모에 유지하고 닫힌 유효 HDF115개를
읽었다. 모델·정규화198개, actor/Q/replay0 고정과 실제 terminal 일치를 확인했다.

| 중형 구역 | 유효 시도 | 양손 모두12cm 이내에 들어온 시도 | 그 상태 관측 행 | 양손 닫힘 행 |
| --- | ---: | ---: | ---: | ---: |
| 중간 왼쪽 | 14 | 2 | 35 | 0 |
| 중간 오른쪽 | 14 | 0 | 0 | 0 |

이 두 구역에서 실제 midpoint로는 양손이 가깝지만 nominal gate가 막은 행은0이었다.
따라서 gate 계산 오류나 jaw 설정 하나가 모든 중형 실패를 설명하지 않는다.
오른쪽은 먼저 안전한 진입 경로가 필요하다. 왼쪽 원래 후보 env96은31관측,
약1초 동안 양손 닫기가 허용됐지만 양손 닫힘은 없었고 왼팔 link4의 랙 충돌로
끝났다. 종료의 최대 힘 링크를 최초 접촉 링크로 해석하지 않는다.
[정확한 gate·관측·첫 threshold·명령 근거](assets/rl_v2_closed_medium_gate_entry_20261008.json).

## 실행하는 진단

`scripts/rl/frozen_bilateral_close_probe.py`는 기존 학습 runner를 재사용한다.
동일한 요청128개·checkpoint·waypoint·원래 제어기에서 **양손의 기존 production
12cm gate가 모두 허용할 때만 두 jaw를 닫는다.** 같은 입력에서 body 목표·명령은
그대로이고, 실행 jaw와 기록한21차원 goal jaw를 함께 바꾼다. 이후 접촉 때문에
관측과 다음 body 동작이 달라질 수 있으므로 전체 궤적이 같다고 주장하지 않는다.

진입 거리만 actor의 관측으로 읽으며 privileged pinch로 닫기나 lift를 결정하지
않는다. 성공·0.25초 유지·8mm clearance·랙10N·장애물5N·낙하10cm, self OFF와
원래 무작위화·물리는 유지한다. 고정된 박스나 성공 상태 reset을 쓰지 않는다.

진단 전용 process-local hook은 종료 시 복구하고 learning·DEV/FINAL·축소된
크기 범위·다른 controller/solver 실험을 거부한다. Manifest·실제 learner report에
`frozen_bilateral_close_probe`와 실제 변경 행 수를 기록한다. 해당 tag가 있는
결과는 원래 waypoint/Q 준비 경로에서 거부해 다른 정책의 성공으로 섞이지 않는다.

직접 실행 시 일반 frozen TRAIN runner 대신 이 스크립트에 원래 입력을 전달한다.
관리자가 실행한 정확한 인자는 고유 실행 폴더의`launch.json.command`에 있다.
직접 실행과 별도 Drive uploader를 연결할 때는[기존 백업 문서](RL_GOOGLE_DRIVE.md)를
따르며 새 인증이나 credential 복사를 하지 않는다. 다음 frozen 옵션을 유지한다.

```bash
--no-training --cpu-workplace-probe --base-waypoint-probe \
--unmeasured-size-workplace-probe --steps 900 --device cpu --learner-device cpu
```

관련18개 검사에서 jaw만 변경, 실행·기록 goal 일치, 원본 입력 보존, 학습·축소
범위 거부, hook 복구와 기존 manifest 경로를 확인했다. 물리 결과는 전체 종료와
원래198개 tensor 검증 후 판단한다. 이 검사가 파지 성공을 증명하지는 않는다.
추가 검사에서 이 진단의 manifest를 원래 waypoint/Q 준비가 실제로 거부하는지도
확인했다(추가1개 통과). Controller·reward·관측에 추가 변경은 없다.

진단으로 중형 파지가 생기면 이후 실제 TRAIN의 jaw 탐색을 arm 탐색 선택과
분리하고 짧은 닫기 유지 구간을 검토한다. 중형 오른쪽처럼 근접 자체가 없으면
닫기 강제나 보상 크기만 바꾸지 않고 안전한 손 진입 탐색을 개선한다.
