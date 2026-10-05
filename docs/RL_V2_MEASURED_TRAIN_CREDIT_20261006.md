# 실제 TRAIN 연속 동작으로 SAC credit 보강: 2026-10-06

더 단단한 동적 flap을 유지한 GPU3 SAC 전체 DEV는6/128→7/128이었다.
중간 왼쪽2/32·오른쪽5/32, 양쪽 위 선반은0/32다. 일반화 성공이나 확실한 개선으로
해석하지 않는다. 기존128-env 학습을 유지하고 같은 물리·관측·동작 계약에서
n-step critic 보조 학습을 별도로 시험한다.
[전체DEV128·안전 원인·고정기준비교](assets/rl_v2_firmer_flap_guarded_SAC_DEV7_20261006.json).

## 관측 증거와 변경 목적

동일한 실제 TRAIN 성공 경로12개의 첫 recorded action에서 평균 min-Q는
Q6864의−0.2075→Q8378의−0.2480, 당시 실제 탐색 행동 경로의 discounted return은
8.3768이었다. 위 왼쪽 동일590행 경로도 첫 Q−0.3238→−0.3659, 관측 return6.8421이었다.
[동일12경로](assets/rl_v2_actual_TRAIN_path_credit_Q6864_Q8378_paired_20261006.json)와
[후속 전체 경로](assets/rl_v2_actual_TRAIN_success_path_credit_after_8378_20261006.json).

![동일 성공 TRAIN의 초기 Q와 위 왼쪽 경로](assets/rl_v2_actual_TRAIN_path_credit_followup_20261006.png)

관측 행동 return은 현재 정책 Q의 unbiased label이 아니다. 차이만으로 critic이
틀렸다고 단정하거나 reward를 수정하지 않는다. Terminal 보상의 fitting과 앞선 행동
credit은 별개이므로 연속 동작 결과를 전달하는 별도 목적을 시험한다.

## 옵션형 n-step 보조 Q

기본 one-step SAC에 가중치0.1의 critic 보조 loss를 더한다. Gamma는0.999이며
최대16 control steps의 실제 reward 합과 실제 endpoint의 target-Q를 사용한다.

```text
n = min(16, 실제 terminal까지 남은 step)
target = sum(gamma**j * recorded_reward[t+j], j=0..n-1)
       + gamma**n * endpoint의 현재 정책 target min-Q
terminal이면 bootstrap = 0
```

Endpoint는 기존19-D bounded body correction와 binary jaw4개 분기의 정확한 기대값이다.
현재 entropy backup은OFF이며 중간 entropy를 reward로 새로 넣지 않는다. Intermediate
action은 수집 당시 행동 정책의 action이다. Importance correction 없는 n-step에는
off-policy bias가 있을 수 있어 one-step을 대체하지 않고 실제 전체 DEV로 판단한다.

- 성공과 유한한 안전 위반 실패를 모두 보관한다. 초기 무효·respawn·numerical quarantine·
  altered-waypoint/solver probe·불완전한 경로는 제외하고 이유를 기록한다.
- 각 next actor/critic 관측이 다음 current 관측과 정확히 같아야 한다. Reset이나
  episode를 넘지 않고 마지막 행만 실제 terminal이어야 한다.
- 지역별8192행/총32768행 이내 whole episode를 유지한다. 성공이 있으면 지역별 최소
  1episode를 보호한다. 가용 지역을 균등하게, 그 안의 실제 행을 균등하게 sample한다.
- Q update마다64행을 보조 loss에 쓴다. Actor loss·탐색·reward·안전·DR은 유지한다.
- Bank는 `staged_goal_experience.pt`에 저장하고 checkpoint에는 설정·개수만 남긴다.
  Frozen eval은 optimizer/replay를 갱신하지 않는다.

원래 actor/Q/target-Q·normalizer·4optimizer·counter·성공 보존 schedule·21-D replay는
그대로 복원한다. 보조 목적은 물리 계약 밖의 `measured_train_credit`에 명시한다.
기본은OFF이며 켜진 checkpoint를 `one-step`으로 조용히 재개하려는 요청은 거부한다.
DEV/FINAL·old VR reward·success-only MC label·inverse-action label을 만들지 않는다.

## 실제 기록 연결과 사용법

[`prepare_measured_train_credit.py`](../scripts/rl/prepare_measured_train_credit.py)는 종료·
Drive 검증 완료된 native run을 matching replay와 연결한다. Compact actor feature를
원본464-D 관측으로 오인하지 않고 critic에 담긴 원본 actor/critic·38D current/next flap·
reward·종료·held clock을 비교해 원래 recorded21-D goal 행만 가져온다.

실제 TRAIN360경로(성공27·실패333)를 연결했고 정확히 연결되지 않는2경로는 제외했다.
용량 제한 후 bank는31,447행/64경로(성공10·실패54)다. 위 왼쪽 성공1경로와 양쪽 상단
실패도 포함한다. 기본 replay224,621행은 변경하지 않았다.
[입력 연결과 bank 출처](assets/rl_v2_measured_TRAIN_credit_preparation_20261006.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  /path/to/env_isaaclab_232/bin/python scripts/rl/prepare_measured_train_credit.py \
  --checkpoint /absolute/path/to/matching/checkpoint_00006864.pt \
  --closed-run /absolute/path/to/closed-native-run \
  --output-dir /absolute/path/to/new-initialization-directory
```

기존 `batched_staged_goal_with_drive.py --gpu 3 --training` 명령의 checkpoint를
새 입력으로 지정하고 `--measured-train-credit measured-nstep16`을 추가한다.
원래 waves/waypoints/training manifest/demo/native source 인수를 유지한다.
실행 폴더는 새 고유 경로이며 `CUDA_VISIBLE_DEVICES=3`으로 격리한다.
기존 Drive 연결·300초 백업·크기/MD5 검증 후 최신2개 보호·writer 종료 후 로그 검증을
유지한다. 새 대용량 replay도 검증 전에는 삭제하지 않는다.

## 진입 각도 진단과 확인

Frozen upper-right DEV4개×yaw0/−5°/+5°의12attempts는 성공0이었다. 원래 각도는
유효4/4, 바뀐 각도는 각각1/4로 초기 무효3개도 각 분모4에 남긴다. 모든 후보가 유효한
seed121303은 원래/−5°에서 랙 충돌, +5°에서 충돌 없이 timeout·contact quality0이었다.
기본 waypoint를 바꾸지 않는다. [닫힌 전체 결과](assets/rl_v2_firm_flap_upper_right_yaw_DEV4x3_20261006.json).

CPU 검사84개를 통과했다. 실제 실패/성공·불연속/reset 차단·terminal decoder 제외·
gamma^16 endpoint·binary jaw 기대값·actor/Q/optimizer 보존·bank 재개를 확인했다.
검사는 물리 성공을 증명하지 않는다. 실제 새 실행과 보조 Q update는 PID·현재
progress/metrics로 확인한다. 원래 box/base/background DR·동적 firmer flap·네 영역32개씩
평가를 유지한다. 독립 FINAL은 아직 사용하지 않았고 goal은 계속 진행 중이다.

02:41:49 KST에 `actual_flap_measured_credit16_sac_pgs128_gpu3_20261006_024149`를 시작했다.
Run은 `batch_sac_20261006_024149_31996b`, 실제 writer3585346·supervisor3585308이다.
서비스와UID·실행 경로·CUDA3를 확인했고 원래 writer2763569는 유지했다.
초기 전체 DEV → original TRAIN7/8 → 후속 전체 DEV의4wave다. 현재 초기화와
첫 평가를 진행하므로 실제 보조 Q 갱신 및 성공률 개선은 아직 확인 전이다.
입력4개는 별도 CPU 서비스가 기존 Drive 연결로 업로드·검증하며 완료 전에는 보존한다.
Notion은 새 native PNG를 포함한95개 미디어를 확인했고 기존94개를 모두 보존했다.

02:52 KST에 GPU0 nominal 대조 학습의 후속 전체DEV는3/128이었다(actor3962/Q17894).
중간 왼쪽1/32·오른쪽2/32·양쪽 상단0/32로 이전4→6에서 다시 낮아져 장기 학습만으로
안정적인 개선을 확인하지 못했다. GPU3 기존 writer는actor2282/Q11176으로 TRAIN을
계속했고 새 n-step 실행은첫DEV91step/Q6864로 아직 optimizer를 갱신하지 않았다.
이 시점의 초기 평가와 새 보조 Q의 향후 학습 효과를 구분한다.

## 03:13 KST: 성공 동작과 현재 jaw 선택 대조

Immutable checkpoint actor2319/Q11322를 CPU에서 읽고 이전 실제 성공 TRAIN15경로의
같은 state에서 recorded goal과 현재 deterministic goal을 비교했다. 중간 왼쪽5경로의
평균 body goal 차이는0.01136, 중간 오른쪽9경로는0.01238, 위 왼쪽1경로는0.00680이다.
이는 normalized19D goal 차이이며 cm나 관절각 오차로 해석하지 않는다.
[같은 성공 상태의 현재 행동 비교](assets/rl_v2_actual_TRAIN_current_policy_success_actions_Q11322_20261006.json).

양쪽 손이 production 접근 gate 안에 있는1,286행 중 recorded/current jaw가 다른 행은
101개다(중간 왼쪽28/384·오른쪽69/688·위 왼쪽4/214). 현재 body를 고정한 네 jaw 조합의
learned min-Q 최고값과 현재 jaw의 차이가0.05를 넘는 행은685개다. Learned Q가 더 높다고
실제로 안전하게 파지할 수 있다는 뜻은 아니며, critic의 privileged 입력으로 배포 동작을
고르는 제어기를 추가하지 않았다.

두 jaw를 함께 바꿔야 더 높은 Q가 되면서 현재 factorized 확률의 두 expected-Q gradient가
그 방향과 반대인 후보는2/1,286행이었다. 이는 네 branch Q의 읽기 전용 국소 진단이며
Bernoulli 구조가 주원인이라는 증거가 부족하다. 새 joint categorical 구조를 지금 도입하지
않고 actor·gripper loss를 유지한다. 위 왼쪽 성공 경로에서도 후반 일부 상태는 Q가 조기
닫힘을 선호했지만 실제 성공 행동은 아직 열림이었다. 같은 경로 마지막8행은 현재·recorded·
최고 Q 모두 양손 닫힘이었다. Q 대조를 물리 반사실 성공이나 올바른 닫힘 시점의 label로
오인하지 않는다. [네 jaw 분기·확률·gradient·단계별 원본](assets/rl_v2_actual_TRAIN_four_jaw_choices_Q11322_20261006.json).

03:12:56의 실제 소유 PID·run 경로·CUDA 대조에서 기존 GPU3는 TRAIN5/361step,
actor2475/Q11946·새 held TRAIN157,772행으로 계속 갱신했다. GPU0 대조도
actor4178/Q18758로 진행한다. 새 n-step GPU3는 초기 DEV0/661step으로 actor1204/Q6864,
replay224,621행·online0을 그대로 유지했다. 초기 DEV 중 optimizer를 갱신하지 않는 것이
의도된 동작이다. 아직 새 보조 Q의 실제 update나 후속 성능을 확인한 단계는 아니다.

새 실행의 reset 기록은18개 physical box asset과 마지막 reset의11개 환경 ID에 대해
PhysX property readback을 통과했다. 강성1.5–2.5·감쇠0.15–0.25·정마찰0.45–0.65·
동마찰0.30–0.40·초기±1° 계약을 확인했으며 이 마지막 기록을 전체128개 reset으로
표현하지 않는다. Dynamic flap과 원래 box/base/background randomization을 유지한다.

새 n-step 초기 입력 manifest/waves/checkpoint/replay4개는03:06:38 KST에 기존 Drive에서
각 크기·MD5 검증까지 마쳤다. 약2.16GiB replay도 포함하며 로컬 원본은 삭제하지 않았다.
이 시점 로컬 여유14.63GiB, 세 실행의 backup error는 없고300초 checkpoint 백업을
계속한다. 목표와 독립 FINAL의 상태는 변하지 않는다.
