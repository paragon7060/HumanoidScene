# V2 grasp: 데모 연결과 학습 진행 요약

2026-10-02: 훈련 배치8/8, 별도의 새 배치 평가8/8에서 양손 flap 파지 유지에
성공했다. 안전 위반·invalid reset·시간 초과는 모두0이다. 최종 actor는7,662
update이며 평가8회는 이 하나의 checkpoint를 고정했다.

현재는 **데모의 기준 제어 + SAC 보정 정책**이다. 독립 SAC의 성공이나 모든
선반·박스 크기·초기 자세에 대한 성공을 확인한 결과는 아니다.

## 데모와 SAC의 연결

![데모 기준 제어와 SAC 보정의 연결](assets/rl_v2_demo_sac_pipeline_20261002.png)

1. VR 성공 데모2개 중 아래 선반 데모를 현재 S63·Leju·upright torso 제어로
   재현해 실제 성공 transition410개를 확보했다. 위 선반 데모는 현재 제어에서
   아직 성공하지 못했다.
2. 기준 경로의 관절·base·torso 목표를 인식된 박스 위치에 맞춰 변환한다.
   경로와 그리퍼 타이밍은 실행 중에도 제공된다.
3. SAC는 현재 로봇·랙·모든 박스 관측과 기준 목표를 보고22차원 보정을 낸다.
   팔 목표의 최대 보정은±0.03rad이다. 기준 제어와 합쳐 실제24차원 명령을 낸다.
4. 실제 물리 실행의 관측·보상·종료 결과만 replay에 넣어 actor/critic을
   학습한다. 데모의 제안 동작을 가짜 성공 전이로 사용하지 않는다.

데모에서 확보한 실제410개 전이는 zero-residual 초기 데이터다. Actor의
zero-residual 초기화 prior는512 updates 동안 사라지지만 기준 제어는 남는다.
모방 초기화가 끝났다는 것과 데모 경로 의존성이 없다는 것은 다르다.

## 측정 결과

| 실험 | 결과 | 해석 |
| --- | --- | --- |
| 이전 일반24-action SAC actor350 | 파지0, 시간 초과 | 독립 정책은 아직 실패 |
| 데모 기준 경로 재현 | 아래 선반 성공 / 위 선반 실패 | 학습 없이 가능한 기준 동작 점검 |
| 다양한 배치 SAC 훈련8회 | 8/8 성공, actor7,662 updates | 실제 보정 학습 중 결과 |
| 같은 모델의 새 배치 평가8회 | 8/8 성공, 추가 optimizer/normalizer update0 | 해당 제한 분포에서의 고정 모델 성능 |
| 같은 배치에서 SAC 보정0 | 현재2/2 성공, 총3회 비교 진행 중 | 학습의 추가 기여는 아직 확정하지 못함 |

성공 범위는 작은 박스·아래 선반 같은 구역·안쪽2–4cm 위치 변화·주변 활성
박스0–3개다. 시작 yaw는±1°지만 롤러에서0에 가까워지므로 큰 yaw 일반화로
해석하지 않는다. 보정0에서도 성공한 두 배치 때문에8/8을 SAC의 개선 효과로
해석할 근거는 아직 부족하다. 기준 제어의 성공과 학습의 기여를 별도로 측정한다.

실험: `artifacts/rl/drive_runs/layout_suite_gpu3_20261002_0440/results.json`.
훈련/평가16회에서 관측된 actor updates와 종료 결과, 각 run의 최종 Drive 검증이
기록되어 있다. 임의의 전체 배치에 대한 통계적 보장은 아니다.

## 실제 새 배치 성공 영상

평가 배치1: 안쪽2.86cm 이동, 주변 활성 박스3개. 고정 모델이411tick(13.7초)에
양손 파지 유지에 성공했다. 실제 GPU3 PhysX 자세를 CPU mesh로 표시한다.
우측 대기 영역의 비활성 박스는 학습의 활성 주변 박스와 구분해야 한다.

[![새 배치의 양손 flap 파지 성공](assets/rl_v2_layout_heldout_success_20261002.png)](assets/rl_v2_layout_heldout_success_20261002_h264.mp4)

[H.264 영상](assets/rl_v2_layout_heldout_success_20261002_h264.mp4) ·
[실제 배치·종료·고정 모델 기록](assets/rl_v2_layout_heldout_success_20261002.json).

## 진행 중인 확대와 남은 문제

- GPU0: 같은 배치3개에서 보정을0으로 만든 물리 비교. 학습하지 않으며
  기존 기준 제어만으로도 성공하는지를 측정한다.
- GPU3: 기존 평가와 백업 완료 후 새 훈련12개·평가12개를 실행한다. 안쪽2–4cm,
  앞뒤±1cm, 시작 yaw±1°, 주변 박스0–3개의 고정 분포다. 모든24개 배치의
  실제 box footprint를 기존2mm 사전 margin으로 검사했다. 물리 정착과 각
  active box의 기존 shelf/region 검사도 유지한다. 첫 훈련 배치는411tick에 성공,
  actor8,358 updates다. 아직 전체 평가 성공률은 없다. 첫 배치의 sampled
  depth는+5.08mm였지만 정착 후 기준 대비 실제 depth 차이는0.0002mm였다.
  따라서 이 성공을 앞뒤 위치 일반화의 증거로 보지 않는다. 롤러 정착이 변화의
  대부분을 없앴는지 실제 물리 pose를 계속 확인한다.
- 위 선반: full-wrist/closing-axis IK, torso 앞쪽4cm, 양손 동시 닫기,
  원본 gripper timing을실제로 점검했다. 각각과 일부 조합이 모두900tick 시간 초과였다.
  초기 flap 상태 등 구 데모에서 복원할 수 없는 상태도 있어 완전한 원본 재생은
  아니다. 좌표계·접촉·관절 제한·시점을 계속 분리해서 확인해야 한다.
- 기준 경로 의존성을 줄이는 독립 정책은 별도로 확인해야 한다. 위 선반도
  실제 성공 데이터를 확보한 후 학습에 연결하며, 실패를 해결되었다고 보지 않는다.

현재 reward, 양손 flap 성공 조건, 랙10N/주변 장애물5N, self-collision 제외와
upright torso는 유지한다. Curriculum을 추가하지 않았다. 새 배치 파일은
`layout_depth_split_20261002_0630`, GPU3 확대 실행은
`layout_depth_gpu3_20261002_0630`, 보정0 비교는
`layout_zero_baseline_gpu0_20261002_0617`이다.

## 영상 형식과 보관

새로 올린7개 영상은 mp4v 코덱이었고 그중2개는 application/mp4 타입으로
일반 파일에 등록되어 있었다. H.264(avc1, yuv420p, faststart)·video/mp4로 변환한
복구본을 기존 Notion 기록에 추가했다. 원본 frame 수·FPS·해상도와 전체 decode를
확인했다. 기존 Drive 원본은 유지한다. 이전 첨부의 직접 교체가 거부되어 새 복구
영상과 아래 요약 페이지를 사용한다. 브라우저 재생 자체를 직접 확인한 것은 아니다.

새 물리 재생의 저장은 writer 종료 → H.264 변환 → 전체 decode 검사 → 원자적
파일 교체 → 완료 상태 기록 순서다. 변환 실패 시 원본을 남기고 실험을 실패로
기록한다. CPU만 사용한다. `scripts/rl/browser_video.py`의 CLI로 이미 종료된
영상도 다른 경로에 변환할 수 있다. 닫힌 원격 원본을 덮어쓰지 않는다.

Targeted CPU tests: **107 passed**. 영상 코덱·timing 보존·변환 실패 시 원본
보존과 배치 depth/footprint 검사를 포함한다. 실제 성공률은 PhysX 결과로 판단한다.
Drive는 기존 인증으로5분마다 검증 업로드하며 최근2개 checkpoint 및 보호된
검증본을 유지한다. 다른 사용자의 파일·프로세스는 변경하지 않는다.

[읽기 쉬운 Notion 하위 페이지](https://app.notion.com/p/3ec63918d42a81389724c8cc53084726) ·
[자세한 일반화 보고서](RL_V2_LAYOUT_GENERALIZATION_20261002.md) ·
[Drive 보관 규칙](RL_GOOGLE_DRIVE.md).
