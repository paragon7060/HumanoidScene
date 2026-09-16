# 2026-09-15 git pull 및 SAC 중력 보상 검토

## 재-pull 결과: 690d43d에서 SAC 공통 제어 연결됨

추가 요청 후 main을 a14c1c3에서 690d43d로 다시 fast-forward pull했다.
이번 커밋은 `Add shared gravity compensation and 1 kg Leju claws`이며,
아래 a14c1c3 기준의 중력 보상 부재 분석 이후 새로 반영된 변경이다.

SAC의 alternatives runner → build_configs → WorkcellRLEnvCfg → RL scene의
build_robot_cfg → configure_robot_asset_physics → configure_gravity_compensation
순서로 `cfg.class_type = GravityCompensatedArticulation`이 설정된다.
ManagerBasedRLEnv의 매 physics step scene.write_data_to_sim 호출은 이 클래스의
_apply_actuator_model을 실행한다. 따라서 JointDeltaTargets에 직접 보상 코드가
추가되지 않았어도 SAC/PPO/DPPO가 모두 공통 writer의 중력 보상을 받는다.

보상 방식은 force drive의 시뮬레이션 전용 목표에 g/Kp bias를 더하는 것이다.
논리 action 목표는 유지하고 기존 drive effort cap 안에서 PD와 합산된다.
Isaac Lab 2.3.2의 부모 writer가 매 호출 목표 버퍼를 다시 채우므로 bias는 누적되지
않는다. S200062의 팔 14개와 torso 4개를 대상으로 하며 head/바퀴/그리퍼/수동
linkage 및 잠긴 관절은 제외한다. fixed root만 지원하고 박스 payload는 보상하지
않는다. 아래 이전 4-env 실험은 별도 external effort 방식이므로 그 미세 오차
수치를 이번 g/Kp 구현의 성능 측정값으로 해석하면 안 된다.

기존 로컬 수정/미추적 파일 50개는
`artifacts/rl/git_sync/repull_20260915_235559`와 named stash에 보존했다.
테스트 파일 2개의 복원 충돌은 로컬의 calibrated EEF fixture와 debug adapter
지원으로 해결했으며 원격 추가 의도도 충족한다. 기존 파일 누락 및 미해결 충돌은
없다. body-lock 테스트의 축약 agent fixture에는 기존 로컬 PPO 설정에 필요한
algorithm 객체를 추가했다. 실제 학습 agent에는 원래 해당 객체가 존재한다.

완료된 SAC 1500 iteration 학습은 eba8e0b 기반의 별도 source snapshot을 사용했으며
그 snapshot에는 이번 공통 writer가 없다. pull은 과거 학습 결과를 바꾸지 않는다.
이번 작업에서 새 학습 또는 GPU 물리 평가를 시작하지 않았다.

## Git 동기화

main을 dbbe8b3에서 a14c1c3까지 fast-forward pull했다. 기존 수정/새 파일 60개를
별도 백업 및 named stash로 보존했다. rewards.py의 추가 함수 위치 충돌 1건은
원격의 guard_box_reward와 로컬 flap_alignment/flap_closing을 함께 보존해 해결했다.
나머지 원격 변경은 자동 병합했다. 이미 main에 반영된 안전 종료 관련 새 파일
4개는 로컬 백업과 byte 단위로 같음을 확인했다. 미해결 충돌 및 누락 파일은 없다.
원래처럼 로컬 변경은 unstaged로 남기고, stash도 복구용으로 보존했다.

주요 원격 변경: S63 공식 자산 및 Leju two-finger variant, Quest base/torso
조작 평활화, rollerless Quest 디버그 최적화, 접촉 보고 주기 제어. 현재 SAC에서
사용한 S200062 integrated gripper preset은 기존 source snapshot과 같다.
새로운 중력 보상 제어는 원격 변경에 포함되어 있지 않다.

백업: `/home/seonho/HumanoidScene/artifacts/rl/git_sync/pull_20260915_232000`.

## 결론

**중력 보상 토크가 없는 기존 PD 제어는 이번 전체 관절 환경의 유력한 문제다.**
실제 동일 PhysX 환경에서 zero-action 유지 시 torso/arm 추종 오차와 손 TCP의
큰 처짐을 확인했다. 이는 reward의 누락과는 구별해야 한다. 현재 task에는
파지 3, lift progress 5, holding 5, success 150의 보상이 이미 있다.
무게/중력 potential energy만 보상하는 항을 추가하는 것으로 제어 문제를 해결할
수 있다고 판단하지 않는다.

## 4-env 실제 물리 비교

학습 때 사용한 원본 source snapshot, named 초기 자세 quest_ready_02,
S200062, 오른손 flap 6cm lift task, 마찰 20/16, gravity ON, fixed base를 사용했다.
Checkpoint 1500은 환경 계약만 확인하고 **policy action은 사용하지 않았다**.
네 환경 모두 zero action으로 180 control step / 6초 유지했다. 아래 값은 마지막
30 step / 1초 평균이며 torso·팔 자세 목표와 실제 관절 차이를 함께 기록했다.

| 설정 | 오른손 TCP 초기 대비 이동 | 높이 변화 | 팔 최대 추종 오차 | torso 최대 추종 오차 |
|---|---:|---:|---:|---:|
| 기존 PD: 팔 220 / torso 400 | 339.9 mm | −331.7 mm | 7.29° | 22.68° |
| 팔 PD만 800으로 강화, torso 400 | 336.2 mm | −291.1 mm | 1.86° | 19.45° |
| 기존 PD + 중력 보상 토크 | <0.1 mm | <0.1 mm | <0.001° | <0.001° |
| 팔 PD 800 + 중력 보상 토크 | <0.1 mm | <0.1 mm | <0.001° | <0.001° |

중력 보상은 root_physx_view.get_gravity_compensation_forces()의 관절 항을
팔·torso·head의 실제 제어 관절 20개에만 매 physics step effort target으로 더했다.
그리퍼 모터/손가락/수동 linkage/바퀴에는 추가하지 않았다. 최대 보상 토크는 약
165.1 Nm였고 모든 환경 reset은 0회였다. 기존 PD의 torso 최대 오차는 leg_joint의
약 22.7°이다. 팔 게인만 올려도 torso 처짐이 남기 때문에 TCP 오차가 거의 그대로였다.

기존 제어의 zero action은 목표 관절각을 유지하지만 실제 관절각을 고정하지 않는다.
PD는 중력 하중을 버틸 위치 오차가 필요하므로 중력에 의한 정적 편차가 생길 수 있다.
PhysX도 feed-forward 중력 보상으로 큰 게인 의존성을 줄이는 방식을 설명한다.
[PhysX joint drive 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html),
[PhysX 중력 힘 계산](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html).

## 학습 부진과의 관계 및 한계

30cm 이상 손 위치가 처지는 동안 파지 후보 접근/정렬/닫기를 동시에 학습해야
하므로, 처음부터 25 action을 탐색하는 SAC에서 부담이 크다는 추론이다.
현재 학습 결과를 중력 보상 하나의 원인으로 확정하지는 않는다. 이 비교는
초기 자세의 정적 유지이며 파지 성공률·학습 수렴 A/B가 아니다.

재개 901~1500 전체 600 iteration에서 성공은 0회였다. 마지막 1401~1500에는
완료 episode 15,287개, 성공 0, 평균 오른손 목표 거리 1.139m,
평균 파지 점유율 약 0.0008%였다. NaN/Inf transition 제외는 0개였다.
파지/후속 lift의 reward가 거의 나오지 않는 문제, 닫기 보조 함수가 현재 reward
manager에 연결되지 않은 점, 움직임/박스 흔들림 비용 회피, 넓은 action space의
탐색은 별도 검토할 필요가 있다.

팔·torso의 중력 보상 토크를 먼저 공통 제어 경로의 옵션으로 넣는 것이 우선이다.
물리 중력은 유지하고, gripper/수동 linkage는 제외하며, torque limit·PD effort와
합산되는 방식 및 floating-base 관절 offset을 검토해야 한다. 이는 부하를 들어올릴
박스의 접촉 힘 자체를 상쇄하는 것이 아니다. 그 다음 동일 reward/seed/action 설정으로
중력 보상 ON/OFF 학습을 비교하고, 파지·lift·목표 거리와 비용을 함께 판단한다.

이번 작업은 pull/검토만 수행했다. 진단에만 feed-forward를 적용했으며,
실제 학습 코드·기본 servo 설정·reward 설정·저장 checkpoint는 변경하지 않았다.
새 학습도 시작하지 않았다. 진단 프로세스는 종료됐고 GPU 3에 남지 않았다.

## 검증 및 재현 파일

pull 후 관련 CPU 회귀 테스트 29개 통과. whitespace 검사 및 미해결 충돌 검사 통과.
진단 결과: `/home/seonho/HumanoidScene/artifacts/rl/diagnostics/gravity_review_20260915_232214/play_20260915_232221_f3b72d/gravity_probe.json`.
실행 명령: `/home/seonho/HumanoidScene/artifacts/rl/diagnostics/gravity_review_20260915_232214/launch.json`.
진단 코드: `/home/seonho/HumanoidScene/artifacts/rl/diagnostics/gravity_review_20260915_232214/gravity_probe_source.py`.
전체 학습은 `docs/SAC_SAFE_RESTART_20260915.md`를 참고한다.

## 추가 확인: 기존 teleop 중력 보상

프로젝트 전체에는 이미 teleop 중력 보상이 있다. `teleop/teleop_ik.py`의
PersistentTeleopIKAction은 2026-09-01 d7d695e5에서 추가된 중력 힘 조회 및
`gravity / stiffness` 위치 bias를 PD 목표에 더한다. 이는 외부 effort를
직접 추가하는 방식과 다르며 기존 PhysX drive torque cap을 유지한다.
Quest teleop 및 RL Quest 디버그의 IK 경로에서 사용한다. 최근 pull에서
새로 추가된 코드는 아니며 SAC/PPO/DPPO의 JointDeltaTargets 경로에는 연결되지 않았다.
따라서 “프로젝트 전체에 중력 보상 코드가 없다”는 뜻으로 해석하면 잘못이다.
앞선 무입력 실험은 RL 경로의 부재를 확인한 것이고, 향후 공통화 시 기존
teleop 위치 bias 방식을 우선 재사용·비교하여 전체 torque cap을 보존할 수 있다.
