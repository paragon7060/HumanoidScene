# V2 grasp SAC: 장기 학습 붕괴와 복구 실험 (2026-10-03)

## 종료된 실험 결과

10/02 초반의 연속 성공은 장기 SAC 학습에서 유지되지 않았다. 예정된 횟수를
모두 수행한 뒤 종료했으며, 런타임 오류나 OOM으로 중단된 실행은 아니다.
아래 결과는 종료된 `results.json`과 각 실제 물리 실행의 `metrics.json`에서 집계했다.

| 실험 | 실제 훈련 | 마지막 모델 고정 평가 | 안전 위반 / 시간 초과 |
|---|---|---|---|
| GPU3 목표 SAC, gamma0.99 | 9/48 성공 | 0/12 성공 | 훈련28/11, 평가11/1 |
| GPU0 목표 SAC, gamma0.999 | 6/48 성공 | 0/12 성공 | 훈련16/26, 평가0/12 |
| GPU0 BC, optimizer 업데이트0 | 훈련 없음 | 10/12 성공 | 평가0/2 |

동일한12개 랜덤 box·초기 base XY/yaw 평가에서 **현재 BC가 마지막 SAC보다 좋다.**
SAC가 BC보다 우월하거나 다양한 크기·위 선반까지 성공했다고 주장하지 않는다.
종료된 세 suite의 최종 Drive 업로드 크기·MD5 검증은 완료됐다.

![실제 학습 붕괴 지표](assets/rl_v2_goal_sac_collapse_20261003.png)

[그림 원본 지표와 물리 실행 출처](assets/rl_v2_goal_sac_collapse_20261003.json).
거리 그래프는 **종료 시 더 먼 손의 거리**이며 에피소드 평균 거리가 아니다.

## 원인 후보와 확인된 사실

GPU3은13,190–18,774 actor update 사이 처음9/9 성공했다. Prior 손실 가중치는
3.41에서0.61로 떨어졌다. 다음 실행이20,384 update에 도달하면서 가중치가0이
되었고, 그 이후39개 훈련은 모두 실패했다. 왼손 거리가16.7cm로 벌어지고 이후
랙 충돌이 반복됐다. 실패 중에도 Q 추정값이 상승했으므로 TD loss 감소만으로
실제 파지 성능이 개선됐다고 판단할 수 없다.

기존에는 실제 데모 replay 혼합과 frozen 신경망 prior 손실을 **같은 schedule로
동시에 제거**했다. 성공 동작 주변에서 측정한 좁은 replay로 학습한 Q gradient가
그 영역 밖으로 actor를 이동시키는 것이 주요 원인 후보다. 감쇠와 붕괴의 시간적
일치는 뚜렷하지만 단독 인과를 확정한 실험은 아니다. Gamma0.999 수정만으로도
실패했으므로 할인율 불일치 하나로 모든 실패를 설명하지 않는다.

## 수정한 학습

`PoseGoalSACPilot`의 새 BC 초기화와 명시적인 checkpoint fork에 다음을 연결했다.

- 실제 데모 replay는 기존처럼20%에서20,000 actor update 동안0%로 감소한다.
- Frozen **신경망**의 현재 관측·clock 출력과의 actor MSE 가중치 floor10을 유지한다.
  실행 중 데모 명령 배열을 공급하지 않으며, 이 손실은 가상의 Q 전이를 만들지 않는다.
- SAC가 생성하는 normalized absolute goal을 neural prior 출력의±0.05 범위로 제한한다.
  실제 범위는 goal scale에 비례한다. 이번 초기 모델은 base XY 약±2.33/2.71mm,
  yaw±0.00840rad, 팔 관절 약±0.015–0.058rad이다. 환경의 관절/충돌 기준은 그대로다.
- 과거에 **실제로 실행한** off-policy action과 reward는 clipping하거나 수정하지 않는다.
  Q 데이터에는 이전 실제 명령을 그대로 보존하고, inverse goal decode 재현 검사를 유지한다.
- 몇 회 학습마다 동일한 개발 배치에서 optimizer를 끈 평가를 한다. 성공 수가 검증된
  최고 actor보다 줄면 그 actor만 복구한다. 최신 critic·critic optimizer·실제 성공/실패
  replay·전체 update count는 유지한다. Actor optimizer moments는 초기화한다.
- 개발 배치는 선택과 진단에 반복 사용하므로 독립 최종 성능으로 보고하지 않는다.
  마지막 평가 seed600–611은 앞선 튜닝에서 쓰지 않은 별도 배치다.

코드: `src/kuavo_isaaclab_scene/rl/multi_box/experiments/pose_goal_sac.py`,
`scripts/rl/fork_pose_goal_sac.py`, `scripts/rl/recover_pose_goal_actor.py`,
`scripts/rl/layout_residual_with_drive.py`.

## 실행과 지속 작업

GPU3의 `goal_guarded_base_box_sac_gpu3_20261003_160950` 실행을10/03 16:11 KST에 시작했다.
Parent와 Isaac child 모두 `CUDA_VISIBLE_DEVICES=3`; 내부 장치는`cuda:0`이다.
새 checkpoint는 기존 성공 actor12,494와 실제 replay7,192개에서 fork했다.
Gamma0.999에 맞추는 critic-only warmup2,000회 후 actor12,494/critic14,994로 시작한다.
실행 전 실제 native seed410개와 모든 goal/replay contract를 확인했다.

24개 랜덤 훈련 배치×2회, 처음 개발 평가4회와 훈련4회마다 개발 평가4회,
마지막 새 holdout12회로 총112개 실제 실행을 예정했다. 첫 기준 고정 평가는414tick에
성공·unsafe/invalid/timeout0, 종료 후 Drive 검증도 끝났다. **학습 전 기준 한 번의
성공**이며 새 온라인 학습의 개선으로 해석하지 않는다. 현재는 정확한 복구 진단을
위한 sequential one-env이며80GB VRAM을 채우는 병렬 학습은 아니다.

박스는 동적 물체이며 안쪽2–4cm·yaw±1°·주변 박스0–3개 randomization을 유지한다.
실제 초기 base는 rack-frame 좌우±20cm·밖으로3–25cm·yaw±15°다. Curriculum,
성공 판정, rack10N/obstacle5N, self-collision off 설정을 변경하지 않았다.
기존 Drive remote를 재사용해5분마다 백업하고, 검증된 오래된 checkpoint만 정리하며
로컬에 최신2개와 형식별 검증된 최신2개를 보호한다. 복구 replay도 업로드·검증한다.

사용자의 성공까지 계속하라는 요청에 따라 Codex의 **지속 목표를 active로 등록**했다.
이 도구는 현재 thread에서 후속 작업을 자동으로 이어가는 기능이다. 현재 연결에는
`automation_update` 예약 도구가 없어 특정 시각의 채팅 예약을 만들었다고 주장하지 않는다.
실행 종료·오류·성능 퇴행을 확인하고 분석·수정·재학습을 계속하며, 목표를 아직 완료로
표시하지 않는다. 다른 사용자 GPU1/2 실행과 파일에는 변경하지 않았다.

## 위 선반과 남은 검증

위 선반의 current-rest IK 진단은900tick 시간 초과였고, 이어진 base6cm 접근 진단은
566tick에`zarm_l4_link`–rack35.51N으로 실패했다. 모두 종료·백업 검증됐다.
따라서 위 선반은 여전히 미해결이다. Base 접근량을 늘리는 것만으로 해결된다고
보지 않으며, 팔 진입 자세와 그리퍼 닫힘 시점을 함께 검토해야 한다. GPU0에서
`upper_coordinated_reach_gpu0_20261003_161948` 진단을 시작했다. 실제 생성된
폴더와 상태로 완료 여부를 확인한다. Base 접근3cm, 양손12mm 안에 들어온 뒤
함께 닫기, current-rest IK와 torso 앞쪽4cm를 비교했다. 이 실행도578tick에 왼팔
`zarm_l4_link`–rack44.41N으로 실패했다. Gripper는 열려 있었고 왼손 목표에2.83cm
IK reach projection이 남았다. 종료·최종 Drive 검증 완료. 접근 거리를 줄이고
닫힘을 늦추는 것만으로 도달 범위·팔 진입 문제를 해결하지 못했다. 이 VR/IK 실행은
SAC가 아니며 실패 데이터에서 가상의 native 성공 seed를 만들지 않는다.

이번 정책 제약·legacy resume·실제 replay 보존·개발 평가 rollback 회귀 검사
**73 passed**, 기존 SAC·replay·Drive·배치·영상 회귀까지 포함해 **130 passed**.
새 제약 적용 후의 파지 성공률은 실제 실행 결과로 추가 보고한다.
