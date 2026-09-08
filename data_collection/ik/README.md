# Task1 IK backend

Task1의 기본 실행기는 IsaacLab DLS를 유지한다. Kuavo와 같은 IK를 비교·사용할
때만 이 폴더의 standalone `plantIK` backend를 선택한다.

## 원칙

- 시뮬레이터는 Kuavo5W의 `src/kuavo_isaaclab_scene/assets/kuavo5/kuavo5.urdf`를
  사용하고, solver에는 같은 URDF에서 arm link/joint만 추출한 외부 runtime용
  adapter를 생성한다. 별도의 손으로 작성한 kinematic 모델은 두지 않는다.
- ROS, LeTools, `kuavo_humanoid_sdk`는 이 경로의 필수 의존성이 아니다.
- Kuavo 저장소에서 가져오는 것은 `plantIK.h`와 `libplantIK.so`뿐이다.
- solver/runtime은 Git repo나 episode output에 넣지 않는다. Kanu의 별도 runtime root
  (기본 예: `/home/seonho/.cache/humanoidscene/kuavo_plantik/<commit>/`)에 둔다.
- backend가 실제 joint 결과를 반환하고 scene FK 잔차를 확인하기 전에는 Task1
  실행기를 이 backend로 자동 전환하지 않는다.

## 구조

```text
data_collection/ik/
├── README.md
├── runtime.yaml                 # 모델/frame/protocol/외부 runtime 위치
├── kuavo_plantik_client.py     # Isaac/Python에서 persistent worker 호출
├── native/
│   ├── CMakeLists.txt
│   └── src/kuavo_plantik_server.cpp
└── scripts/
    ├── fetch_kuavo_plantik.sh  # pinned header/.so만 외부 runtime에 설치
    ├── make_kuavo5w_arm_urdf.py # scene URDF에서 14-DoF arm adapter 생성
    └── build_kuavo_plantik.sh
```

`native` worker는 한 번 모델을 로드한 뒤 stdin 한 줄마다 다음 28개 값을 받는다.

```text
q0[14], left_pos[3], left_quat_xyzw[4], right_pos[3], right_quat_xyzw[4]
```

`q0`/반환 `q`는 Kuavo5W arm 순서(`zarm_l1..7`, `zarm_r1..7`)이다.
stdout에는 `1 q[14]`(성공) 또는 `0`(실패)을 한 줄로 반환한다. 따라서 Isaac
simulation loop 안에서 매번 프로세스를 새로 띄우지 않는다.

## 준비/실행

```bash
cd /home/seonho/HumanoidScene
export KUAVO_IK_RUNTIME=/home/seonho/.cache/humanoidscene/kuavo_plantik/1797481d4639a76afbdc035f705acbf522fceafa
data_collection/ik/scripts/fetch_kuavo_plantik.sh "$KUAVO_IK_RUNTIME"
python data_collection/ik/scripts/make_kuavo5w_arm_urdf.py \
  --source src/kuavo_isaaclab_scene/assets/kuavo5/kuavo5.urdf \
  --output "$KUAVO_IK_RUNTIME/model/kuavo5w_arm.urdf"
data_collection/ik/scripts/build_kuavo_plantik.sh "$KUAVO_IK_RUNTIME"
```

빌드는 Drake CMake 패키지(`drake_DIR` 또는 `DRAKE_DIR`)를 요구한다. ROS master가
없어도 되지만, Drake ABI가 `libplantIK.so`와 맞아야 한다. 첫 smoke는 Kuavo5W
`MediumBox_0` pregrasp target에 대해 q를 받고, 기존 Isaac FK로 양손 위치·방향
잔차를 기록한다. 성공 전에는 `task1_box_pick_web.py`의 기본 DLS 경로를 변경하지
않는다.
