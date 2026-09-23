# HumanoidScene 문서 목차

루트 [README](../README.md)는 첫 실행과 자주 쓰는 명령만 다룬다. 아래에서
현재 목적에 맞는 문서를 선택한다.

실물 S63에 VR 수집 궤적 또는 RL action을 안전 제한과 함께 배포하는 코드는
[`real_robot_control/`](../real_robot_control/README.md)에 독립적으로 정리했다.

## 처음 시작

1. [설치 및 첫 실행](INSTALL.md)
2. [Isaac Sim Workcell 편집](ISAACSIM_WORKCELL_GUIDE.md)
3. [Quest 빠른 시작](QUEST3_QUICKSTART.md)

## 코드 개발

- [코드 구조와 개발 위치](CODE_STRUCTURE.md): scene/env, Quest, robot/asset,
  display, recording, evaluation 경계와 Python 경로 변경 안내
- RL 장면·병렬 환경은 `rl/scenes/`와 `rl/envs/`, manager 설정과 계산은
  `rl/managers/`와 `rl/mdp/`, 기존 robustness 환경은
  `envs/manager_env.py`와 `envs/manager_mdp.py`에서 수정한다.
  Standalone 화면 구성과 실행은 `envs/scene.py`다.

## Scene과 asset

- [Isaac Sim Workcell 편집](ISAACSIM_WORKCELL_GUIDE.md): prim 구조, 위치·회전·크기,
  rack-relative box pose 캡처와 respawn
- [Gripper 구성](GRIPPER.md): 패키지 구조, host별 장착, 접촉 모델, pad 모델, TCP, 수정 위치
- [Gripper 설정](GRIPPER_CONFIGURATION.md): S56 QiangNao/S200062 two-finger 선택과 외장 Leju/Robotiq 구성
- [Third-party assets](../THIRD_PARTY_ASSETS.md): 외부 asset 출처와 라이선스
- [독립 Leju claw asset](LEJU_CLAW_ASSET.md): 좌우 URDF/USD, 재생성, 장착 원점과 재사용 API

## Meta Quest

| 상황 | 읽을 문서 |
|---|---|
| 처음 연결한다 | [Quest 빠른 시작](QUEST3_QUICKSTART.md) |
| 실제 수집기 SDK·인증서부터 준비한다 | [수집기 설치·간편 실행](QUEST_COLLECTOR_SETUP.md) |
| 이미 설치된 서비스를 다시 실행한다 | [Quest Runtime 실행](QUEST_RUNTIME_SERVICE.md) |
| 조작키·보정·카메라·데이터 schema가 필요하다 | [Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md) |
| 관찰자 화면과 성능 옵션을 정한다 | [Quest 화면·성능](QUEST3_DISPLAY_AND_PERFORMANCE.md) |

Quest 문서의 역할은 다음과 같이 구분한다.

- 빠른 시작: 반드시 필요한 실행 순서와 첫 HDF5 수집
- Runtime 실행: CloudXR service, 웹 서버, 인증서와 터미널별 재실행
- 상세 가이드: controller/hand mapping, episode 제어, LeRobot schema, 세부 문제 해결
- 화면·성능: VR 화면, wrist panel, 선택적 head sensor, PC observer viewport와 GPU 부하

## RL 학습

루트 README의 [RL 구현·사용법 요약](../README.md#10-rl-구현과-사용법-요약)에서
시작하고, 목적에 따라 다음 문서를 읽는다.

| 상황 | 읽을 문서 |
|---|---|
| 지금 한 손 flap 집기를 학습·재개·평가한다 | [1단계 flap pick](RL_FLAP_PICK.md) |
| 전신 4박스 작업을 단계별 / 전체 직접 학습으로 비교한다 | [4박스 전신 RL 실험](RL_MULTI_BOX.md) |
| Multi-box v2 grasp SAC를 짧게 검증한다 | [Multi-box v2 SAC pilot](RL_MULTI_BOX_V2_PILOT.md) |
| 오른손만 학습하고 양쪽 flap 면의 거리·파지·disturbance를 수정한다 | [오른손 파지 상세와 튜닝](RL_RIGHT_HAND_PICK.md) |
| Quest로 직접 움직이며 VR에서 항목별 reward를 확인한다 | [Quest reward 검사](RL_QUEST_REWARD_DEBUG.md) |
| Quest로 V2 grasp SAC와 같은 형식의 시연 전이를 기록한다 | [Quest RL 시연 수집](RL_QUEST_REWARD_DEBUG.md) |
| VR 없이 GUI에서 양손의 네 손가락 기준점을 맞추고 저장한다 | [데스크톱 파지 기준점 보정](GRASP_DESKTOP_CALIBRATION.md) |
| 보정한 TCP 중심, RL/IK/기록 적용 범위와 VR 표시를 확인한다 | [공통 endeffector_center](ENDEFFECTOR_CENTER.md) |
| 일반 환경과의 차이, 병렬 복제·좌표·충돌 격리를 수정한다 | [배경 없는 RL 병렬 환경](RL_PARALLEL_ENVS.md) |
| `quest_ready_02` 관절/base 값을 수정하거나 VR로 다시 캡처한다 | [RL 초기 상태](RL_INITIAL_STATES.md) |
| 다른 하위 task, observation/action/reward 또는 PPO를 수정한다 | [하위 task별 RL 학습](RL_TRAINING.md) |
| SAC / diffusion 사전학습 / DPPO를 실행한다 | [대체 RL 학습 경로](RL_ALTERNATIVES.md) |
| 오른팔 / 전체 관절 action space를 선택한다 | [공통 RL action 옵션](RL_ACTION_SPACES.md) |
| SAC settling 데이터 필터와 curriculum 없는 닫기 보상 변경을 확인한다 | [SAC 데이터·보상 수정](SAC_DATA_REWARD_20260909.md) |
| 체크포인트를 Google Drive에 업로드하고 로컬 보관량을 줄인다 | [Google Drive 보관](RL_GOOGLE_DRIVE.md) |

각 문서의 역할은 실행·task 정의, 장면·병렬 구조, 초기 자세, 범용 manager 개발로
구분한다. 공통 폴더와 import 경계는 [코드 구조 가이드](CODE_STRUCTURE.md)에 있다.

### Quest로 RL reward debugging

저장소 루트에서 기존 Quest 수집기와 같은 연결 절차를 사용한다.
Runtime/web이 이미 켜져 있으면 앞의 두 명령은 다시 실행하지 않는다.

```bash
# 터미널 1: CloudXR Runtime
./quest_collector.sh runtime
# 터미널 2: Quest 접속 페이지
./quest_collector.sh web
# Quest에서 기존 HTTPS 페이지에 접속하고 CONNECT → 터미널 3
./quest_collector.sh collect --rl-reward-debug \
  --rl-config configs/rl_pick_arms_only.py
```

X로 시점을 보정하고 A로 시작/정지한다. 오른손 검지 트리거로 gripper를 조작하며,
Y로 reward 패널 표시/숨김, B로 초기 자세 복원 후 새 시도를 준비한다.
체크포인트 없이 직접 움직이며 검사한다. 이 명령은 데이터 수집이나 학습을 수행하지
않는다. 일반 수집기와 동시에 실행하지 않는다. 값 없는 `--rl-reward-debug`는
legacy 단일 박스 flap-pick 검사이며, `--rl-reward-debug 2`는 multi-box v2
진단 장면을 검사한다. V2 grasp SAC 환경에서 시연 전이를 저장하려면
`--rl-reward-debug 2 --rl-demo-dataset <새 HDF5 경로>`를 사용한다.
명령 전체와 데이터 schema는 [Quest RL 시연 수집](RL_QUEST_REWARD_DEBUG.md)에 있다.

패널에는 실제 가중치·제어시간이 적용된 항목별 step reward, 합계/누적 return,
파지·상승·유지 상태와 부족한 성공 조건이 표시된다. 종료 시에는 자동 reset 전의
마지막 값이 남는다. 카메라 영상이 필요 없으면
`--no-quest-camera-overlay --no-camera-preview`를 추가한다.

- 표시 항목·이름·순서·소수점: `src/kuavo_isaaclab_scene/rl/debug/reward_report.py`
- 새 상태값 수집: `src/kuavo_isaaclab_scene/rl/debug/reward_recorder.py`
- 패널 위치·크기·글꼴: `src/kuavo_isaaclab_scene/display/xr_reward_panel.py`
- 실제 reward 가중치: `configs/rl_pick_arms_only.py`의 `configure` 또는
  `src/kuavo_isaaclab_scene/rl/managers/rewards.py`
- 실제 reward 계산식: `src/kuavo_isaaclab_scene/rl/mdp/rewards.py`

표시 수정 예시와 단위·계산 주의사항은
[디버깅 패널 수정 방법](RL_QUEST_REWARD_DEBUG.md#디버깅-패널-표시값-수정)에 있다.
수정 후 검사 프로세스를 재시작해야 반영된다.

## Policy와 평가

- [GR00T N1.7 평가](GROOT_N1_7_EVAL_GUIDE.md)
- [VR 자세 저장과 eval 초기 상태](INITIAL_STATES.md): 이름 있는 자세 캡처·복원,
  GR00T 몸체 고정과 RL 초기 상태 안내
- [RwH-Kuavo V2 GR00T N1.5를 S56에서 평가](RWH_KUAVO_V2_S56_EVAL.md):
  별도 Conda worker, 16-D arm/claw 변환, headless 3-view MP4, VRAM과 배포 설정
- [로봇 모델 평가 파이프라인](ROBOT_MODEL_EVAL_PIPELINE.md): 공통 계층, 호환성
  경계, 새 USD/URDF·gripper·checkpoint profile 온보딩 체크리스트
- [Offline policy 평가](OFFLINE_POLICY_EVAL.md)

## 상세 참조

- [기존 scene 구성, task-system과 검증 기록](../PROJECT_REFERENCE.md)

문서의 `/absolute/path/...`와 IP 주소는 예시다. 현재 PC의 실제 경로와 LAN IP로
바꿔서 사용한다.

## S63 / Leju claw

공식 S63 URDF의 claw 포함 여부와 현재 지원 상태는
[S63 / Leju claw 가이드](S63_LEJU_CLAW.md)를 참고한다.
Robotiq 2F-85는 제거했다. S63은 기본 `none`, donor claw는
`--gripper leju-twofinger`로 선택한다.
