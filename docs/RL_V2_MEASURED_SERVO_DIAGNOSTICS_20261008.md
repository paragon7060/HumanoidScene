# SAC 실제 수집 상태의 속도 제한 진단

## 현재 성능과 목적

2026-10-08 20:38 KST 기준, 성공 동작은 있지만 안정적인 SAC 개선은 없다.
관절 목표를 URDF 범위로 정규화한 비교 실행은 초기14/128 → TRAIN384 뒤8/128 →
TRAIN768 뒤4/128이다. 마지막 평가는 성공4·안전 위반49·시간 초과75회이고
초기 무효·수치 오류는0이다. 안전 위반은랙 충돌48·박스 낙하1회다.
중형과 중간 오른쪽small·상단 왼쪽small은 성공0이다. Actor1,909·Q9,682회
실제 모델·optimizer와 평가 종료 모델의 일치·유한성을 대조했다.
[전체 결과와 모델 근거](assets/rl_v2_URDF_second_learned_full_DEV8_20261008.json).

팔 목표를 전체 URDF 범위에서 표현하는 새 full-arm SAC는 학습 전 전체 평가에서
성공11·안전 위반48·시간 초과56·초기 무효13/128회였다. 초기 무효도 분모에
포함한다. 실제 저장 모델·normalizer75개 tensor는 준비한 초기 모델과 같고
전체257개 tensor는 유한했다. Actor/Q 업데이트0인 **별도 초기 기준**이다.
다른 실행의14회와 똑같은 접촉 이력이 아니며 학습 개선으로 해석하지 않는다.
[새 정책의 전체 초기 결과](assets/rl_v2_URDF_full_arm_first_full_DEV_20261008.json).
기존 네 장기 실행은 그대로 진행한다. 독립FINAL은 미사용이고 목표는 미달성이다.

![원래 여섯 조합 DEV128의 결과: 성공14→8→4회. 무효·안전 위반·시간 초과를 포함하며 중형은0회다.](assets/rl_v2_URDF_whole_DEV14_8_4_20261008.png)

## 최신 학습 모델의 실제 실패 영상

[중간 왼쪽small](assets/rl_v2_URDF_DEV8_middle_left_small_rack_failure_20261008.mp4)과
[중간 왼쪽medium](assets/rl_v2_URDF_DEV8_middle_left_medium_rack_failure_20261008.mp4)은
TRAIN768 뒤 전체 평가에서 기록한 실제 자세 영상이다. 둘 다랙 충돌 실패이고
양손 유지 시간은0초다. Small은 오른쪽 그리퍼 base89.2N, medium은 왼쪽 그리퍼
base28.7N이 종료 순간의 최대랙 접촉이었다. 전체 첫 충돌의 원인을 영상만으로
단정하지 않는다. 실제 terminal pose를 reset 전에 기록했고 actor1,909·Q9,682
모델과 전체 평가 결과에 맞는다. H264/avc1·yuv420p·faststart 및 전체decode를
확인했다. 이 overlay의Q9,682는 Q **업데이트 횟수**이며 Q-value가 아니다.
[영상·결과·형식 검증](assets/rl_v2_URDF_DEV8_closed_video_verification_20261008.json).

## 확인된 문제와 아직 확인할 부분

절대 관절 목표를 실제 명령으로 바꾸는 production decoder에는 관절 속도 제한이
있다. 목표가 한 제어 step의 이동 범위를 넘으면 명령은 최댓값으로 제한된다.
그 구간에서는 목표를 조금 바꿔도 실제 명령이 그대로여서, servo Q를 통한 해당
목표의 경사가0이 될 수 있다. 팔 목표 범위를 넓히는 것만으로 이 문제는 해소되지 않는다.

새 정책의 실제 초기 모델을 **종료된 다른 Cartesian 진단 제어기의 TRAIN 경로**에
질의했다. 유효114경로에서 초반·중간·후반 각912개 상태를 사용한 결과다.

| 구간 | 팔 좌표 중 step 제한으로 경사0 | 팔14좌표 모두 경사0인 상태 |
| --- | ---: | ---: |
| 초반 | 27.1% | 1/912 |
| 중간 | 48.4% | 149/912 |
| 후반 | 71.7% | 379/912 |

이는 현재 SAC가 만든 상태의 측정이 아니다. 현재 학습 실패의 원인으로 단정할
수 없으며 물리 성공이나 탐색 경로 개선도 증명하지 않는다. 상태 창은 겹칠 수
있어 독립 episode 수처럼 세지 않는다. 모델·입력·RNG는 변하지 않았고 진단
행동·평가 데이터는 학습에 넣지 않았다.
[정확한 초기 모델·닫힌 입력·경사 측정](assets/rl_v2_full_arm_initial_closed_TRAIN_servo_gradients_20261008.json).

## 이번 코드 변경

- 두 CPU 경사 감사 도구가 실제 state-aware Gaussian mean과 실제 noise/offset
  sampler를 사용하도록 수정했다. Full-arm의 학습 residual mean을 그대로
  Gaussian mean으로 취급하면 기준 관절 목표를 잘못 재구성한다.
- 닫힌 경로의 actor/Q 복원은 접근 전 snapshot 대신 실제 첫held 제어 행의
  박스 좌표를 고정 기준으로 사용한다. 접근 중 박스가 움직일 수 있기 때문이다.
  Live `BatchedBaseStages`와 같은 시점이며 학습 환경의 동작은 바꾸지 않는다.
  이전27.1/48.4/71.7% 분석도 이 좌표로 다시 계산했고 집계 비율은 같았다.
- `audit_closed_train_servo.py --by-box-type`으로 같은 구역의small/medium을
  별도 집계할 수 있다. 기본 출력은 유지한다.
- `train_batched_staged_goal.py --policy-servo-diagnostics`를 추가했다.
  실제 실행에 쓰는 pre-action actor518D 입력에서 greedy 및 실행 행동의
  속도 step 제한 비율을30step마다 구역·박스 크기별로 기록한다.
- 이 옵션은 기본OFF다. 정책·reward·Q·replay·RNG를 바꾸지 않는다. 기존 실행에는
  소스를 교체하지 않는다. 재사용은 새 실행에만 적용한다.

진단을 켠 새 실행에는 기존 학습 명령에 다음 옵션을 추가한다.

```bash
--policy-servo-diagnostics
```

현재 표본은`progress.json.policy_servo_diagnostics`, 종료 배치는
`policy_servo_diagnostics_wave_####.json`에 기록된다. 한 표본은 상태가 남아 있는
환경만 포함하므로 전체128조건의 성공률은`metrics.json`으로 따로 판단한다.
측정은 **step 제한에 의한 경사0**만 다룬다. 전체 actor/Q 경사, tanh 수치
포화 또는 실제 접촉 품질 전체를 측정한 것으로 해석하지 않는다.

기존 full-arm 장기 실행의 첫 원래 TRAIN128조건과 같은 요청으로 별도 짧은
수집을 준비했다. 실제 CPU 복원에서 actor·normalizer29개 및 동결 기준이 같고
계약 차이는 replay100,000 대2,000,000뿐이다. 새 Q/replay/성공 은행에서 시작한다.
900step의 최대Q1,800회는 actor warmup2,048회보다 작으므로 초기 정책의 실제
수집 상태를 확인할 계획이다. 실제 실행·종료 counter로 다시 확인해야 한다.
이 진단은 장기 학습이나 원래 전체DEV/독립FINAL을 대체하지 않는다.

20:43 KST에 실제 GPU3 전용 writer959568로 이 측정을 시작했다. 고정 소스는
`9990e5e15171ea1d534893c5329ec0ebdb8faf58`이고 옵션을 실제 명령에서 확인했다.
20:51 KST에 실제held 관측117개·진단 출력과 Q118회 수집을 확인했다. Actor
갱신은0이다. Actor518/critic578·body19+jaw2·원래첫TRAIN128요청·소스735개와
중력 보상18관절·기존 물리/성공/보상/안전/DR 계약을 대조했다. 입력257개 tensor는
유한했으며 실제 종료 모델의 일치는 종료 뒤 확인한다. 첫 Drive 백업은20:49:28 KST
검증됐고 실제 writer는1,539MiB를 사용했다.

이 **step121의 한 표본**에서 greedy 팔 좌표의 step 제한 비율은 전체32.7%, 실제
수집 행동은20.6%였다. 구역/크기별greedy 비율은 중간 왼쪽small9.7%·medium19.6%,
중간 오른쪽small39.3%·medium18.8%, 상단 왼쪽small40.0%·오른쪽small48.3%다.
일부 좌표에서 경사 차단이 실제 SAC 상태에도 존재함을 확인했지만, 전체 경로의
비율·최종 실패 원인·학습 후 성능을 뜻하지 않는다. 움직이기 위해 정상적으로
속도 제한에 도달한 좌표도 포함한다. 종료한 경로 전체와 손 접촉·Q 근거를
함께 분석한 뒤 제어 표현 변경을 판단한다.
[실제 시작·계약·단일 관측·백업 검증](assets/rl_v2_measured_TRAIN128_actual_startup_20261008.json).

관리 폴더는`GPU3_URDF_full_arm_measured_TRAIN128_20261008_204327`이고 기존 네 장기
실행은 유지했다. 이 짧은 실행은 성능 개선을 주장하거나 오래 학습하는 실행을
대체하지 않는다.

이 측정의 CPU 관찰기는 실제 writer와 supervisor가 모두 종료되고 마지막
Drive 체크포인트·로그 검증이 끝난 뒤에만 HDF를 읽는다. 원래128요청·실제
저장 모델/optimizer·초기 actor 유지·끝난 경로 전체의 경사를 함께 대조할 계획이다.
현재 단일 표본과 닫힌 전체 경로 결과를 구분한다.

## 검증과 다음 판단

관련28개 CPU 테스트가 통과했다. 실제 mean 재구성, 정확한 sampler와 첫held 좌표 사용,
크기별 집계와 모델/RNG 보존을 확인했다. 테스트 통과는 파지 성능 증거가 아니다.

현재 SAC 수집에서도 팔 경사가 많이 차단되는지 먼저 확인한다. 많이 차단된다면
원래 물리 속도 제한 안에서 실제 명령을 직접 표현하는 정책 설계와 탐색 단위를
검토한다. 과거 direct-action 실험의 회귀 원인도 함께 확인하며, Q/replay와
계약을 새로 분리하고 동작 기준을 실제 물리에서 비교한 뒤 적용한다. 차단이
적다면 Q 추정·성공 동작 유지·안전 진입 탐색을 우선 분석한다.

박스·base·배경·firm dynamic flap 무작위화, 여섯 구역/크기 배분,
양손 opposing pad5N·0.25초 유지·8mm 들기, 랙10N·장애물5N·drop10cm,
selfOFF·중력 보상18관절·upright torso 설정을 유지한다.
Drive는 기존 연결과300초 업로드·검증된 최신2개 보존을 사용하며,
raw HDF/replay는 로컬에 남고 종료 로그 검증은 writer가 멈춘 뒤 진행한다.
