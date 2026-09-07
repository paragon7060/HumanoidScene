# HumanoidScene 문서 목차

루트 [README](../README.md)는 첫 실행과 자주 쓰는 명령만 다룬다. 아래에서
현재 목적에 맞는 문서를 선택한다.

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
- [Gripper 설정](GRIPPER_CONFIGURATION.md): S56 QiangNao/S200062 two-finger 선택과 외장 Leju/Robotiq 구성
- [Third-party assets](../THIRD_PARTY_ASSETS.md): 외부 asset 출처와 라이선스

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
- 화면·성능: VR 화면, head/wrist panel, PC observer viewport와 GPU 부하

## RL 학습

루트 README의 [RL 구현·사용법 요약](../README.md#10-rl-구현과-사용법-요약)에서
시작하고, 목적에 따라 다음 문서를 읽는다.

| 상황 | 읽을 문서 |
|---|---|
| 지금 한 손 flap 집기를 학습·재개·평가한다 | [1단계 flap pick](RL_FLAP_PICK.md) |
| 일반 환경과의 차이, 병렬 복제·좌표·충돌 격리를 수정한다 | [배경 없는 RL 병렬 환경](RL_PARALLEL_ENVS.md) |
| `quest_ready_02` 관절/base 값을 수정하거나 VR로 다시 캡처한다 | [RL 초기 상태](RL_INITIAL_STATES.md) |
| 다른 하위 task, observation/action/reward 또는 PPO를 수정한다 | [하위 task별 RL 학습](RL_TRAINING.md) |
| SAC / diffusion 사전학습 / DPPO를 실행한다 | [대체 RL 학습 경로](RL_ALTERNATIVES.md) |

각 문서의 역할은 실행·task 정의, 장면·병렬 구조, 초기 자세, 범용 manager 개발로
구분한다. 공통 폴더와 import 경계는 [코드 구조 가이드](CODE_STRUCTURE.md)에 있다.

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
