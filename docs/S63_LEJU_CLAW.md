# S63 / Leju claw 지원 상태

## 공식 파일에서 확인한 내용

확인 기준: `LejuRobotics/kuavo-ros-opensource` beta,
`e2da287383c643e923e9594dac2ec86024269737` (2026-09-15).

- [기본 S63 URDF](https://github.com/LejuRobotics/kuavo-ros-opensource/blob/e2da287383c643e923e9594dac2ec86024269737/src/kuavo_assets/models/biped_s63/urdf/biped_s63.urdf)는
  평탄한 robot 정의이다. 별도 claw URDF include, 손가락 링크,
  개폐 관절, mimic 정의가 없다.
- Gazebo URDF와 MuJoCo XML에도 claw 개폐 관절은 없다.
  Gazebo Xacro는 기본 Gazebo URDF와 depth camera Xacro를 포함할 뿐이다.
- [mesh 폴더](https://github.com/LejuRobotics/kuavo-ros-opensource/tree/e2da287383c643e923e9594dac2ec86024269737/src/kuavo_assets/models/biped_s63/meshes)에
  `l/r_twofinger.STL`, D405 및 hand-pitch 파일이 존재해도,
  mesh만으로 articulated claw가 생성되지는 않는다.
- 공식 저장소에는 Leju claw SDK/드라이버가 있지만, 이것은 실제 장치 제어
  인터페이스이며 시뮬레이터의 기구학·관성·접촉 모델을 대신하지 않는다.

## 현재 프로젝트

- Robotiq 2F-85 preset, 변환 단계, packaged asset을 제거했다.
  Robotiq를 Leju claw로 이름만 바꿔 사용하지 않는다.
- `s200062` / `s200062_integrated` two-finger rig도 명시적으로 선택할 수 있다.
  원본 기반 형상에 물리 four-bar closure와 추정 손 관성을 보완한
  시뮬레이션 구성이다. 제조사 관성 보정값으로 간주하지 않는다.
- `s56_twofinger`의 기존 donor rig도 유지했다.
- `s63`는 기본 `none`이다. 외장 hand articulation, gripper action 또는
  자동 Robotiq 대체가 생성되지 않는다.
- `--gripper leju-twofinger`를 명시하면 S200062-derived 독립 claw와 D405를
  합친 별도 S63 variant를 선택한다. 원본 mesh를 수정하지 않고, 양손 구동 및
  four-bar closure와 접촉 설정을 연결했다. 이는 공식 S63 claw 모델이 아니며
  mesh 겹침과 파지 성능은 사용자가 확인해야 한다. [사용법](LEJU_CLAW_ASSET.md).
- S63 폴더를 위 커밋의 공식 `biped_s63` 전체 파일로 교체했다.
  원본 `urdf/biped_s63.urdf`는 그대로 보관한다. 실행용 `kuavo_s63.urdf`는
  `package://.../meshes/` 경로를 `../meshes/`로 바꾸고, 기본 S63 손목 visual은
  `l/r_hand_pitch.STL`을 사용한다 (`*_noHand.STL` 대신 선택).
  EEF는 `xyz="0 0.0 -0.17"`, `rpy="0 0 0"`이며 다른 값은 동일하다.
  실행용 S63 USD도 이 URDF로 다시 생성했다. Isaac용 fixed base 및 joint drive
  설정은 변환 단계에서 적용되며 공식 ROS 파일 자체의 설정은 아니다.
- 기존 virtual wrist camera pose는 검사 용도로 유지하며, Leju claw
  장착 위치·시야에 대한 하드웨어 보정으로 간주하지 않는다.
- Leju variant의 손목 visual은 `l/r_hand_pitch_wrist_only.STL`을 사용한다.
  원본 S63 CAD에서 카메라와 브래킷 3개 부품만 제거한 파생 mesh이며,
  원본 STL과 기본 S63 모델은 유지한다. variant에는 Leju D405 마운트 하나만
  남고, wrist camera 센서는 해당 `l/r_d405_camera` body를 사용한다.
- 기존 Robotiq checkpoint/dataset은 다른 claw와 호환된다고 가정하지 않는다.
  preset 및 state/action 계약이 달라지므로 같은 데이터 경로에 섞지 않는다.

## 실행

2026-09-15 왼손 실측 끝단 폭을 바탕으로 `leju-twofinger.left` 매핑을 보정했다.
완전 열림은 앞/뒤 driver −0.363811/+0.363811 rad이며, 닫기·열기의 서로 다른
폭 곡선을 사용한다. 정적 간격 보정으로 범위를 제한하고,
[측정값·보정 전후 비교·재현 절차](../real_robot_calibration/s63_gripper/README.md)를 기록했다.
오른손에도 사용자 요청으로 왼손 매핑을 동일하게 적용했다. 독립 오른손 실측에
의한 보정은 아니며, 채택한 매핑의 Isaac CUDA 왕복 간격 검증은 통과했다.
닫기 명령 75는 양손의 전체 닫기/중간 역전 경로를 각각 3회 재측정한 결과
25~27 mm로 반복돼, 최초 38 mm 대신 대표값 26 mm를 사용한다.
[반복 측정 결과](../real_robot_calibration/s63_gripper/reports/check75_01.md).

```bash
# 현재 제공하는 two-finger claw
./run_scene.sh --robot-model s200062 --gripper s200062_integrated

# 공식 S63 형상 검사: 외장 gripper를 추가하지 않음
./run_scene.sh --robot-model s63 --gripper none

# S200062-derived claw를 장착한 별도 S63 variant
./run_scene.sh --robot-model s63 --gripper leju-twofinger
```

원본을 갱신한 뒤 실행용 파일을 재생성할 때:

```bash
bash scripts/prepare_s63_urdf.sh
# 전체 로봇 변환이 아니라 S63만 변환
conda activate env_isaaclab_232
python .external/IsaacLab-v2.3.2/scripts/tools/convert_urdf.py \
  src/kuavo_isaaclab_scene/assets/kuavo_s63/urdf/kuavo_s63.urdf \
  src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/kuavo_s63_fixed.usd \
  --fix-base --joint-stiffness 400 --joint-damping 40 --headless --device cpu
```

## S63에 기능하는 Leju claw를 추가하려면

공식 claw URDF/USD 또는 검증된 기구학 자료가 필요하다:
링크별 mesh, 개폐축·링크 길이·joint limit, 연동/폐루프 관계,
장착 transform, TCP, 질량·관성, 접촉면 및 카메라 extrinsic.
현재 S200062 donor rig를 S63에 이식하는 방법도 가능하지만,
실제 S63 Leju claw와 같다는 증거 없이 공식 모델로 표기하지 않는다.
지금 제공하는 donor variant는 위 명시적 옵션으로만 선택한다.
