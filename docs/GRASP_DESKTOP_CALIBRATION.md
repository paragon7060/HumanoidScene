# 데스크톱에서 양손 파지 기준점 보정

Quest를 착용하거나 연결하지 않고, Isaac Sim의 일반 Viewport에서 양손의 손가락 기준점
총 4개를 맞추는 별도 도구입니다. VR 실행 화면에서 편집할 필요가 없습니다.

## 실행

기존 Isaac Lab 및 RL 의존성이 설치된 저장소 루트에서:

```bash
bash scripts/calibrate_grasp_desktop.sh
```

기존 Quest/Isaac Sim 프로세스는 먼저 직접 종료하세요. 이 도구는 다른 프로세스를
중단하지 않습니다. 두 시뮬레이터를 동시에 실행하면 GPU 메모리를 추가로 사용합니다.
CloudXR 서비스나 헤드셋 연결은 필요하지 않습니다.

기본적으로 `configs/rl_pick_arms_only.py`의 `quest_ready_02` initial pose와 RL 장면을
읽되, 하나의 환경만 만들고 초기화 후 **Isaac Sim timeline도 Pause**합니다.
센서 카메라, XR, policy 실행, 학습, 데이터 수집을 시작하지 않습니다.
원래 config 파일과 initial pose 파일은 변경하지 않습니다.

## 점 이동과 저장

1. `Grasp reference calibration (desktop)` 창에서 원하는 손가락의 **Select**를 누릅니다.
2. **Focus**는 해당 점 근처로 데스크톱 카메라를 이동합니다. 이후 일반 Viewport 탐색으로
   각도를 바꿀 수 있으며, HMD 움직임은 화면에 영향을 주지 않습니다.
3. Viewport에 포커스를 두고 **W** 또는 툴바의 이동 도구를 선택합니다.
   이동 핸들로 기준점을 안쪽 접촉면의 원하는 끝 지점에 맞춥니다.
4. 나머지 세 점도 반복합니다. GUI에 손가락 기준 로컬 XYZ(m)가 표시됩니다.
5. GUI의 **Save all 4 points**를 누릅니다. `Saved all 4 points`와 저장 경로를 확인합니다.
   잘못된 좌표나 파일 쓰기 오류는 `NOT SAVED`로 표시되고 기존 파일은 보존합니다.

| 버튼 이름 | 대상 | 색상 |
|---|---|---|
| `l_f_finger` | 왼손 f 손가락 | 주황 |
| `l_b_finger` | 왼손 b 손가락 | 분홍 |
| `r_f_finger` | 오른손 f 손가락 | 청록 |
| `r_b_finger` | 오른손 b 손가락 | 파랑 |

직접 Stage에서 선택할 때는 `/Visuals/GraspCalibration/<finger>/tip`을 선택합니다.
**`tip`의 위치만** 수정하세요. 부모 `<finger>`는 손가락 좌표계이고, 자식 `marker`는
표시용 구체입니다. 실제 로봇 링크/collision이나 이 부모·자식을 옮기는 것은 저장 대상이 아닙니다.
로컬/월드 이동 도구 모두 사용할 수 있으며 저장 시에는 손가락 기준 로컬 XYZ로 기록됩니다.
회전·스케일은 저장하지 않습니다.

저장 파일이 없으면 점은 손가락 링크 원점에 있습니다. 정확한 손끝이 아니고 내부에 가려질 수
있으므로, 화면에서 구체를 클릭하기보다 GUI의 Select/Focus로 선택하고 이동하세요.
이 도구에서는 물리를 재생하지 않습니다. 툴바 Play를 누르지 마세요.
Stop은 초기 상태 복원으로 편집 상태에 영향을 줄 수 있으므로 사용하지 마세요.

## 저장 파일과 다시 불러오기

기본 파일은 저장소의 `configs/grasp_reference_points.json`입니다.
같은 명령으로 다음에 실행하면 자동으로 불러옵니다. Save를 누를 때 해당 파일을 갱신하며,
저장하지 않고 종료하면 편집은 사라집니다. 별도로 USD를 저장할 필요는 없습니다.

별도 보정 파일을 사용하려면:

```bash
bash scripts/calibrate_grasp_desktop.sh \
  --points-file configs/grasp_reference_points_trial2.json
```

기존 설정과 다른 자세는 `--config configs/my_pick_config.py`로 지정할 수 있습니다.
기본 config의 `INITIAL_STATE`는 고정되어 있으므로 다른 자세가 필요하면 복사한 config에서
수정하세요. `--robot-model` 등 공통 RL CLI를 지원하지만 flap pick이 지원하는 로봇/그리퍼여야 합니다.
로봇 모델이 다른 저장 파일은 불러오기를 거부하므로 모델별로 별도 파일을 사용하세요.

## 기존 코드와의 경계

- 신규 진입점: `rl/debug/calibrate_grasp_desktop.py`와 전용 shell script.
- 기존 학습·평가·Quest launcher는 이 도구를 호출하지 않습니다.
- 기존 `GraspCalibration`의 비물리 마커 및 파일 형식을 재사용합니다.
- 파일 저장 이외에 asset/initial pose/RL config를 수정하지 않습니다.
- S200062 integrated hand는 저장 좌표를 다음 실행 때 공통 TCP/파지 계산에 읽습니다.
  실행 중인 학습/VR의 정의는 바뀌지 않습니다. 저장 후 소비 프로세스를 재시작하세요.
  [공통 EEF 정의와 VR 확인](ENDEFFECTOR_CENTER.md)을 참고하세요.
- 같은 파일은 기존 VR의 `--rl-grasp-calibration` 표시 모드에서도 불러올 수 있습니다.

코드 문법과 저장 로직 수준으로만 검토했으며, 실제 GUI/시뮬레이션 동작 확인은 사용자가 진행합니다.
