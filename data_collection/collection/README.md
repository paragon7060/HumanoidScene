# 데이터 수집·저장

## 수집 흐름

설정/seed 기록 → reset·settling → expert 실행 → 결과 검증 → 에피소드 종료·기록 검증 → 다음 시도.

실패 시도도 seed·실패 원인·단계별 결과를 남긴다. 성공 시연 선택과 평가 seed 선택은 분리한다. Pregrasp 진단 영상/trajectory는 pick 성공 시연으로 분류하지 않는다.

## 저장 계약

| 구분 | 저장 대상 |
|---|---|
| 관측 | Head/좌우 wrist RGB, joint position/velocity, hand state, 실제 TCP pose, robot root pose |
| 실행 명령 | `action.ee_target`(좌우 pose 14D), `action.arm_joint_target`(실제 J_arm), `action.hand_target`(실제 G_cmd) |
| GT | 활성 box pose/world·robot frame/velocity, target ID, grasp reference/pose, flap 상태, phase, contact/grasp, box 변화량, success |
| 진단 | 계획 joint target, 실제 target 변경 여부, 구간 최종 EE 목표, 요청 EE pose(존재 시), 실패 사유 |
| 메타데이터 | seed, asset/robot/hand/planner/config identity, dim names/order, 단위·frame·TCP·quaternion·sampling 의미 |

EE action은 매 control step 실제 전달 joint target의 FK로 정의한다. 최종 waypoint나 측정 관절 상태를 action label로 대체하지 않는다. Hand command와 state 차원은 같다고 가정하지 않는다.

GT 저장과 policy 입력은 별개다. 학습 loader는 명시적 feature allowlist로 GT 누출을 방지한다. 미지원 센서는 측정 0으로 채우지 않고 validity/missingness를 기록한다. Box instance ID/순서를 보존하고 가변 개수는 padding+valid mask 등 명시적 schema로 처리한다.

## 10Hz와 포맷

학습용 저장 주기는 10Hz다. Physics/control을 10Hz로 낮춘다는 뜻은 아니다. 명령 전 관측 o_t와 해당 구간의 실제 명령 u_t, 다음 관측/결과를 구분한다. Camera/state/action timestamp와 갱신 주기를 보존하고 raw control 기록과 10Hz resampling을 구분한다.

LeRobot v3를 목표로 기존 recorder/worker를 확장한다. HDF5 원본 병행 여부, raw 고주파 기록 범위, v3 직접 기록 또는 변환 경로는 writer 호환성·I/O 검증 후 확정한다. 현재 이 설정만으로 v3 자동 수집을 실행할 수 있는 상태는 아니다.

Writer가 표준 `meta/info.json`을 생성하도록 하고 vector dtype/정수 shape/실제 dim names를 검사한다. `meta/feature_semantics.json`과 기존 episode sidecar에 의미·provenance를 저장한다. Resume은 dtype/shape뿐 아니라 dim 순서·단위·frame·robot/hand/schema도 비교해 불일치를 거부한다.

## 수집 전 결정/검증

카메라 해상도, 에피소드 수, raw/depth/segmentation 보관 범위, 출력 run 이름은 `../configs/collection.yaml`에서 설정한다. 랜덤화 범위는 `../configs/task1.yaml`에서 설정한다. 코드에 이 값을 고정하지 않는다.

모든 산출물은 `/home/work/mntvol/data/outputs/<run_name>/` 아래 저장한다. 데이터·영상·로그·진단 결과와 실행에 사용한 YAML 사본을 함께 보관한다. Repo에는 코드·설정·문서만 두며 생성 산출물은 넣지 않는다. 기존 run은 덮어쓰지 않고, 명시적인 resume일 때만 schema/config 일치를 검사해 이어간다.

소수 에피소드에서 영상 decode, frame/state/action 개수·시간 정렬, dim names, 결측 처리, LeRobot 읽기, 중단/재시작을 확인한 뒤 대량 수집한다. 수집 코드·실행 명령·이 검증들은 아직 구현 전이다.
