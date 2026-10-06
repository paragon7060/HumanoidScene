# 단단한 flap · CPU/GPU 물리와 SAC 중간 보고

같은 기존 GPU 학습 actor1204의 전체 원래 DEV128 재생에서 안전 양손 파지와
proof lift는 CPU30/128·GPU4/128이었다. **CPU에서 새로 학습한 정책의 성과가
아니다.** 상단오른쪽은 두 실행 모두0/32이며 독립 FINAL도 사용하지 않았다.

![같은 기존 정책의 실제 전체 평가](assets/rl_v2_frozen_backend_whole_pair_20261006.png)

## CPU/GPU 물리가 다른 이유

PhysX는 GPU에서 broad phase·contact generation·constraint solver를 별도로
구현한다. [NVIDIA GPU rigid bodies 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/GPURigidBodies.html).
같은 backend에서도 actor 삽입 순서·scene 구성 등이 재현성에 영향을 주며 다른
platform 간 bitwise 동일성을 보장하지 않는다.
[NVIDIA determinism 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/RigidBodyDynamics.html#enhanced-determinism).

이번 실행에서 확인한 사항과 아직 구분되지 않은 요인은 다음과 같다.

| 항목 | 확인 결과 | 해석 |
| --- | --- | --- |
| 정책 | 각 실행의 network/normalizer178개 tensor 불변 | 학습 업데이트가 차이를 만든 것은 아님 |
| 초기 rack/support | Actual root 위치10µm 이하·roller q/v 동일 | 요청값뿐 아니라 실제 상태를 확인 |
| Asset 속성190개 | 154개 동일,18개 pool의 stiffness/damping36개 다름 | Flap 범위는 같지만 추첨 값은 다름 |
| GPU 접촉 버퍼 경고 | 검사한 닫힌 로그에 overflow 관련 일치0 | 버퍼 부족이 원인이라는 증거 없음 |
| Constructor/contact 이력 | Bitwise 일치시키지 않음 | Backend만의 인과 효과 또는 엔진 결함 미확정 |
| Clone collision filtering | IsaacLab2.3.2가 CPU에는 별도 filtering 호출, GPU에는 env ID 사용 | 초기화 처리 경로도 같지 않음 |

[실제 속성·로그 검사 JSON](assets/rl_v2_CPU_GPU_physics_property_audit_20261006.json).
GPU 버퍼는 CPU처럼 모두 동적으로 증가하지 않으며 부족하면 접촉이 누락될 수
있지만, 그 일반적 가능성을 이번 원인으로 판정하지 않는다.
[IsaacLab2.3.2 PhysX 설명](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg).

## 바꾼 설정과 학습 방법

Flap stiffness1.5–2.5Nm/rad·damping0.15–0.25Nm·s/rad·static friction0.45–0.65·
dynamic friction0.30–0.40·시작 각도±1°를 사용한다. Episode 중 고정하지 않으며
box/base/background randomization은 유지한다. 양손 서로 다른 flap의 실제
접촉/안정 유지와8mm roller-clearance proof lift가 성공 조건이다.
Rack10N·robot-only obstacle5N·self-collision OFF는 유지한다.

새 CPU_PhysX_v1 계약으로 CPU 물리·GPU3 SAC 학습을 분리했다. 기존 actor의
actor/normalizer9개 tensor만 그대로 옮기고 새 Q/replay·성공 bank·critic
normalizer·네 optimizer를 만들었다. 초기 actor/Q update0·replay0을 확인했다.
기존 GPU Q/replay와 CPU DEV 성공30건을 새 학습에 넣지 않는다.
[구현·초기화·검증 기록](RL_V2_FIRM_FLAP_INITIAL_GUARD_20261006.md).

새 실행은128env,새 TRAIN256개,초기/학습 후 같은 DEV128이다. Box/base/background
분포와 네 구역32개 분모를 유지하며 DEV world snapshot은 가져오지 않는다.
새 CPU 학습 성과는 앞으로 닫힌 실제 TRAIN/DEV 결과로 구분한다.

## 영상 수집과 보관

`--eval-video-env-indices 0 5 42 3 2 4`로 전체128개 평가를 유지하면서 중간 좌우·
위쪽 좌우·성공/충돌/timeout 후보6개를 녹화한다. 후보의 과거 성공 여부를 이번
성공으로 가정하지 않으며 새 실제 outcome을 함께 저장한다. 초기 baseline과
CPU 학습 후 정책은 source checkpoint·update 수·물리 장치로 명확히 구분한다.

영상은 종료/reset 직전 실제 body/link pose를 CPU mesh renderer로 그린다.
Actor 관측으로 flap 움직임을 추정하거나 물리를 다시 재생하지 않는다.
닫힌 개별 video writer만 H.264/avc1·yuv420p·faststart MP4로 변환하고 전체
decode를 검사한다. 기존 Drive 인증과300초 checkpoint 검증/최근2개 보호·
종료 후 로그/HDF/replay 백업을 재사용하며 영상·실제 판정 JSON도 보관한다.
Notion에는 외부 Drive 공유 링크 대신 native video/image로 첨부한다.
