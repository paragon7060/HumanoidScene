# 실물 Quest VR 앱의 명령·상태 기록

사용자는 H12 controller의 VR 모드에 진입한 뒤 Meta Quest 3의 알 수 없는 출처 앱 `kuavo_hand_track_MR`로 실물을 조작한다. 이 앱은 프로젝트의 Isaac VR 수집기와 별도 경로다. 보정의 우선 경로는 **기존 실물 VR 조작 → 실물 명령/상태 읽기 전용 기록 → 동일 실물 입력의 sim 재생·비교**다. Isaac VR에서 실물 후보를 생성하는 이전 도구는 보조 경로이며 이 실물 VR 기록을 대체하지 않는다.

## 확인된 코드와 실행 상태

로봇 작업공간은 `$KUAVO_ROBOT_WORKSPACE`로 지정한다. H12의 `robot_state/ocs2_before_callback.py`와 `multi_before_callback.py`는 `LAUNCH_VR_REMOTE_CONTROL_CMD` 환경변수를 읽고 VR launch를 시작한다. `config/h12_vr_launch.yaml`을 적용하고 현장 override는 `~/.config/lejuconfig/h12_vr_launch.yaml`이다. 읽기 검사 당시 override는 없었다.

표준 Quest launch는 `noitom_hi5_hand_udp_python/launch/launch_quest3_ik.launch`다. 2026-09-16 연결 기록에서는 `/monitor_quest3`, Python `/ik_ros_uni`, `/humanoid_quest_control_with_arm`, `/quest_torso_height`, `/quest_turn_180`이 활성화됐다. `/use_cpp_incremental_ik=false`, `/wheel_ik=true`, `/control_torso=false`, `/single_hand_mode=true`, `/reset_joint_to_default=true`였다. 네 노드가 `/kuavo_arm_traj` publisher로 등록돼 있었지만 실제 기록 메시지는 모두 `/ik_ros_uni`에서 왔다.

팔의 확인된 경로는 `/leju_quest_bone_poses` 및 `/quest_joystick_data` → `/ik_ros_uni` → `/kuavo_arm_traj` → `/joint_cmd` → `/sensors_data_raw`다. 몸통은 실제로 joystick으로 조작됐고, `/quest_torso_height`가 `/torso_open_loop_state`를 읽고 `/cmd_torso_vel`을 발행하는 별도 경로가 확인됐다. 다음 기록부터 `/cmd_torso_vel`, `/torso_open_loop_state`, `/vr_whole_torso_ctrl`도 저장한다. 최초 연결 기록에는 이 세 원시 몸통 topic이 없지만 하위 `/joint_cmd`와 `/sensors_data_raw`의 몸통 4관절은 남아 있다.

## 기록 방법

1. 기존 정상 조작 절차로 controller VR 모드와 Quest 앱을 연결한다. Isaac 수집기는 필요하지 않다.
2. 먼저 연결 상태의 읽기 기록으로 입력 경로를 확인한다. 이 확인을 위해 의도적으로 팔을 움직일 필요는 없다. 사람이 컨트롤러를 완전히 고정해야 하는 검사는 아니다.

```bash
cd /path/to/HumanoidScene
REAL_VR_LOG="$PWD/real_robot_calibration/s63_body_arms/data/real_vr_$(date +%Y%m%d_%H%M%S)/connection.jsonl"
python3 real_robot_calibration/s63_body_arms/calibrate.py record \
  --vr-inputs --duration-s 15 --log "$REAL_VR_LOG"
echo "$REAL_VR_LOG"
```

`record`는 명령 publisher·모드 서비스를 생성하지 않는다. `prepare`, `run --send`, 경쟁 VR 노드 종료가 필요하지 않다. 이 recorder는 기존 VR 앱의 움직임을 제어하거나 중단하지 않으며 연결이 끊겨도 기존 VR 앱은 별도로 동작한다. 기존 실물 조작 경로가 움직임과 정지를 담당한다.

기존 기록에는 `/joint_cmd` q/v/요구 torque/gain, `/sensors_data_raw` q/v/추정 vd/torque, `/imu`, quick 목표, MPC/filtered/WBC 목표가 포함된다. `--vr-inputs`는 Quest bone/joystick, arm trajectory/origin/filtered, torso 및 task-space 입력을 추가한다. 관절 입력은 최대 100Hz, pose류 최대 50Hz, Quest bone/joystick은 최대 20Hz로 저장한다. 원본 ROS 필드·publisher·관절 이름·header/receipt/device 시간을 유지한다.

현재 Python `ik_ros_uni.py`의 `/kuavo_arm_traj`, `/kuavo_arm_traj_origin`, `/kuavo_arm_traj_filtered` 값은 degree다. `/joint_cmd`와 센서 q/v는 rad/rad/s다. 입력의 `arm_joint_1..14`를 실물 팔 순서에 대응시켜야 한다. task-space pose를 관절 action과 같은 벡터로 취급하지 않는다. 정확한 활성 publisher 확인 전에는 자동 변환하지 않는다.

종료 시 `vr_record_coverage`가 받은 입력별 개수와 누락 topic을 표시한다. Python/C++에 따라 쓰지 않는 topic도 있으므로 모든 목록이 채워져야 하는 것은 아니다. C++ 공유메모리 전송에서는 ROS arm trajectory가 생략될 수 있다. shadow publisher나 debug 모드를 자동으로 켜지 않는다. `summary ok=true`는 기록 종료 성공이고 움직임·시간 정렬·보정 품질을 보장하지 않는다.

## 현재 검증 결과와 다음 단계

실제 Quest 연결 상태의 15초 읽기 기록이 `summary ok=true`로 종료됐다. `/kuavo_arm_traj` 916개, filtered 910개, `/joint_cmd` 419개, `/sensors_data_raw` 407개를 확보했다. header 시간 기준 팔 VR 목표→motor target 최적 정렬은 전체 약 189ms/RMSE 0.79°, motor target→측정은 약 15ms/RMSE 0.14°였다. 이 수치는 연결법 확인 중 자유 동작 한 번의 파형 정렬 결과다. 제어기 필터·clipping·서로 다른 기록률이 포함되므로 실제 전송 지연이나 모터 지연으로 확정하지 않는다.

이 연결 기록에서 팔 목표 폭은 관절별 7.16–58.48°, 측정 최대 속도는 119.46°/s였고 joystick으로 조작한 몸통 4관절도 움직였다. 마지막 3초에도 최대 약 15.39°의 팔 위치 폭과 51.68°/s가 있어 정지 시험이 아니다. recorder는 명령 publisher나 모드 서비스를 만들지 않았으며 로봇 코드를 수정하지 않았다.

단위/타임스탬프 보존·rate 제한과 분석기를 포함한 보정 테스트 65개가 통과했다. 분석 결과는 `real_robot_calibration/s63_body_arms/data/real_vr_connected_20260916/response_analysis_v3.json`에 있다.

연결법 확인은 완료됐다. 보정용 다음 기록은 바퀴 이동·물체 접촉을 제외하고 기존 정상 조작 범위에서 팔 구간과 몸통 joystick 구간을 구분해 짧게 라벨링한다. 데이터가 부족한 관절/자세만 기존 소각도 시험으로 보완한다. 원본 실물 입력의 sim importer는 아직 구현되지 않았다. sim actuator 비교에는 먼저 `/joint_cmd` q/v를 재생해 sim q/v와 실측 q/v를 비교한다. `/kuavo_arm_traj`는 실물 controller/filter 구간을 별도로 맞추는 입력으로 사용한다. 이전 `analyze_screen.py`는 phase가 있는 소각도 시험용이고 이 자유 동작 로그에 그대로 적용할 수 없다.

실제 WBC의 가속도 PD와 모터 register gain은 Isaac의 torque 기반 stiffness/damping과 직접 호환되지 않는다. 먼저 제어기 필터/지연과 관절 응답을 구분해 전체 응답을 비교한다. q/v만으로 질량·COM·마찰·gain을 각각 확정하지 않으며 torque 단위 확인과 추가 조건 검증은 별도로 수행한다.

## 구분 동작 기록 결과

`data/real_vr_guided_20260916_retry01/record.jsonl`의 60초 기록은 정상 종료됐다. 팔 control mode가 `2`인 8.15–28.03초에 VR 팔 목표가 motor target으로 적용됐고, mode `1` 구간에는 기본 팔 목표로 전환됐다. 따라서 전체 기록을 한 번에 맞추지 않고 8.4–27.8초만 팔 활성 구간으로 분석한다. 이 구간의 VR 목표→motor target 정렬은 약 197ms/RMSE 1.462°, motor target→측정은 약 13ms/RMSE 0.154°였다. 최초 연결 기록의 189ms/15ms와 비슷하게 재현됐다.

몸통 joystick의 `/cmd_torso_vel` 비영 입력은 29.27–46.94초였다. 29.2–47.2초 구간의 몸통 motor target→측정 정렬은 전체 약 27ms/RMSE 0.073°였다. `/cmd_torso_vel` linear.z는 약 −0.0408~0.0491, angular.z는 약 −0.0738~0.0786 범위였다. 몸통 4관절의 측정 폭은 각각 약 4.08°, 9.30°, 5.12°, 5.27°였다.

팔 측정 최대 속도는 183.38°/s였고 관절별 목표 폭도 17.18–116.36°이므로 더 큰 실물 동작은 필요하지 않다. 기록 당시 `/useVrArmKpKd=false`였으며 실제 `/joint_cmd`의 팔 gain은 non-VR `ruiwo_kp/kd`와 일치했다. 별도 `vr_ruiwo_kp/kd` 설정은 이번 제어에 적용되지 않았다. 자세한 결과는 `real_robot_calibration/s63_body_arms/reports/guided_vr_calibration_20260916.md`에 있다.

동일 `/joint_cmd` q/v를 S63+leju-twofinger sim에 재생했다. 기존 팔 `effort_limit_sim=100 Nm` 공통값은 실물 기반 값이 아니므로, 실물 S63 `joint_peak_torque_limits`를 PhysX의 관절별 순간 hard clamp로 적용했다. 팔 1–7의 cap은 각 side `[66.67, 75, 57, 75, 14.1, 14.1, 14.1] Nm`다. 몸통도 knee/leg `668 Nm`, waist pitch/yaw `267 Nm`, head `200 Nm`로 적용된다. nominal/continuous 한계는 별도 thermal/continuous 모델 입력으로 남긴다.

S56은 기존 upstream URDF 기반 관절별 제한을 유지하고, S200062는 packaged upstream MuJoCo XML의 `actuatorfrcrange`를 사용하도록 바꿨다. 따라서 지원 robot preset의 팔에는 더 이상 공통 100 Nm cap이 유효값으로 남지 않는다. wheel과 gripper의 별도 100 설정은 해당 actuator의 독립된 제한이다.

몸통 replay는 관절별 sim-real RMSE 0.070–0.154°로 맞았다. 최초 팔 replay의 관절별 0.70–22.42° 결과는 아래에 설명한 workcell 접촉 오염 때문에 actuator 보정값으로 사용하지 않는다. 활성 controller가 WBC `output_tau_`를 `/joint_cmd.tau`에 넣고 q/dq/kp/kd와 함께 mode 2로 전달하며, `/joint_cmd.tau_max=18`은 `hardware_settings.max_current`이므로 18 Nm cap으로 사용하지 않는다는 소스 판정은 그대로 유효하다.

초기 후속 replay는 compact workcell에 남은 rack/box collider와 자유공간 VR 궤적이 비의도 접촉해 결과가 오염됐다. 같은 고동작 8초 구간이 workcell에서는 pooled 19.570°였지만 robot-only에서는 0.741°였다. 따라서 workcell에서 얻은 WBC/gain 비교 수치는 폐기했다.

robot-only 전체 21.33초 replay에서 physical-cap `PD+G`의 팔 pooled RMSE는 0.706°, worst 1.241°, 포화 0%였다. 기존 G를 live inverse-dynamics로 대체하면 0.652°/0.985°로 개선됐고, 별도 저속 자세에서도 pooled 0.149°에서 0.133°로 개선됐다. 몸통은 0.104°에서 0.237°로 악화되므로 몸통에는 적용하지 않는다. 기록 WBC tau가 G를 대체하는 `recorded-total`은 1.951°/4.013°로 악화됐고 `G+기록 WBC tau`는 중력 중복 가능성이 있어 채택하지 않는다. S63 기본 `auto`는 팔 전용 live inverse-dynamics와 몸통 gravity-PD를 사용한다. full QP는 기본 runtime에 넣지 않는다. replay 기본 장면은 robot-only다. [부하와 최종 판정](../real_robot_calibration/s63_body_arms/reports/wbc_load_and_fidelity_20260916.md).
