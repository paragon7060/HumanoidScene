# HumanoidScene 문서 목차

루트 [README](../README.md)는 첫 실행과 자주 쓰는 명령만 다룬다. 아래에서
현재 목적에 맞는 문서를 선택한다.

## 처음 시작

1. [설치 및 첫 실행](INSTALL.md)
2. [Isaac Sim Workcell 편집](ISAACSIM_WORKCELL_GUIDE.md)
3. [Quest 빠른 시작](QUEST3_QUICKSTART.md)

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

## Policy와 평가

- [GR00T N1.7 평가](GROOT_N1_7_EVAL_GUIDE.md)
- [RwH-Kuavo V2 GR00T N1.5를 S56에서 평가](RWH_KUAVO_V2_S56_EVAL.md):
  별도 Conda worker, 16-D arm/claw 변환, headless 3-view MP4, VRAM과 배포 설정
- [로봇 모델 평가 파이프라인](ROBOT_MODEL_EVAL_PIPELINE.md): 공통 계층, 호환성
  경계, 새 USD/URDF·gripper·checkpoint profile 온보딩 체크리스트
- [Offline policy 평가](OFFLINE_POLICY_EVAL.md)

## 상세 참조

- [기존 scene 구성, task-system과 검증 기록](../PROJECT_REFERENCE.md)

문서의 `/absolute/path/...`와 IP 주소는 예시다. 현재 PC의 실제 경로와 LAN IP로
바꿔서 사용한다.
