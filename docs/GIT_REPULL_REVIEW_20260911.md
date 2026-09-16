# 2026-09-11 재 pull 검토 — f93e9e9

`main`을 `d22dec5` → `f93e9e9`로 fast-forward했다. 새 커밋은
`Reward alignment and lift progress with one-shot grasp credit`다.
기존 로컬 수정은 보존했다. 문서 충돌 1건을 해결하고, 로컬 보상 테스트의 fixture에
새 `flap_progress` 상태를 추가했다. 보상 설계와 과제 설정 자체는 원격 코드 그대로다.

## 이전 문제의 해결 여부

| 항목 | 결과 |
|---|---|
| 미접촉 정렬 상태에서 계속 양수 | 정지 시 0인 정렬 점수 변화량 보상으로 수정 |
| 단순 접촉/선반 위 파지로 계속 양수 | 접촉 보너스는 최초 유효 파지에 +3 한 번, stable_grasp는 비활성화 |
| 목표 높이 미만에서 잡고 버티며 lift 보상 | lift를 파지 중 높이 변화량으로 수정, 정지하면 0 |
| 초기 대기 관측의 0 나누기 | timeout=0일 때 대기 시간으로 나누도록 수정; 유한 관측값 테스트 통과 |
| 랙/주변 물체 충돌 제한 | 미해결: `collision_constraints_enabled=False`, `obstacle_contact_force=20.0` 유지 |
| 반대손을 움직여 받치기 | 미해결: `active_arm="right"` 유지; 왼팔·왼손 고정 |
| 이전 checkpoint 호환성 | 호환되지 않음: 이번에도 이력 관측 11개와 보상 계약 추가 |
| PPO 정규화/탐색 표준편차 | 관련 수정 없음 |
| FCL 거리 미분 테스트 실패 | 동일 실패 유지 |

## 추가로 확인된 보상 반복 획득 경로

아래는 실제 production `FlapProgress.advance()`에 CPU 상태열을 입력한 결과다.
물리 시뮬레이션에서 정책이 이 행동을 할 수 있다는 실험 결과는 아니다.
표시한 값은 해당 보상 항만의 가중 합이며, 시간·흔들림·action 등 비용을 제외한다.

### 상승 → 놓기 → 재파지 사이클: lift +2.5씩 반복

`src/kuavo_isaaclab_scene/rl/mdp/flap_progress.py:60`은 연속 파지 중에만
높이 차이를 계산한다. 놓으면 height=0, lift_ready=False로 바뀌고, 다음 파지는
기준만 갱신한다. 따라서 0cm에서 파지한 뒤 다음 상태열이 가능하다.

1. 파지한 채 3cm 상승(목표 6cm의 0.5): lift +2.5.
2. 파지를 잃음: lift 0.
3. 미파지 상태로 선반의 0cm로 복귀: lift 0.
4. 0cm에서 재파지: lift 0, 최초 파지 보너스도 0.

이 사이클을 3회 입력했을 때 각 회차 +2.5, 총 +7.5였다. 최초 파지 보너스가
재지급되지 않는 것은 정상이나 상승 보상은 재생성된다. 새 최대 도달 높이에만
보상하거나, 미파지 하강에서도 기존 높이 진척을 잃도록 보상 정의를 검토해야 한다.

### 가까이서 정렬 개선 → 멀리서 정렬 악화 → 복귀: orientation +0.3375

정렬 변화량에 현재/직전 거리 중 큰 값으로 계산한 근접 가중치를 곱한다.
거리 1cm에서 alignment를 0.5→1로 개선하면 +0.3375다. 각도를 유지하며
11cm로 이동하고, 그곳에서 1→0.5로 악화시키면 가중치 0으로 비용이 없다.
각도를 유지한 채 1cm로 돌아와도 정렬 보상은 0이다. 원래 상태로 돌아왔지만
해당 항의 합계는 +0.3375다. 한 쌍의 역동작에는 대칭적이지만 거리까지 바뀌는
사이클의 순이득은 차단하지 못한다. 거리·정렬을 함께 포함한 진척 지표나
에피소드 내 최고 진척 기준을 검토할 수 있다.

재현 결과: `/tmp/humanoid_repull_reward_cycles.json`.

## 검증 및 작업 상태

- 전체 CPU 테스트: **538 passed, 1 failed, 1 skipped**.
- 실패는 기존 `test_distance_gradient_agrees_with_fcl`: 0.06763215 대 0.02919251.
  직전 검토에서 pull 이전 코드로도 동일하게 재현한 실패이며 이번 커밋은 해당 코드를 변경하지 않았다.
- skip은 CPU 환경에 ffmpeg가 없는 비디오 인코딩 검사.
- 로그: `/tmp/humanoid_repull_review_pytest.log`.
- `CUDA_VISIBLE_DEVICES=''`로 실행했다. Isaac Sim/GPU 학습 및 기존 프로세스는 변경하지 않았다.
- 로컬 소스 백업: `/tmp/humanoid_repull_20260911_rw4qbmyb`.
  autostash `79e7eb5`도 복구용으로 유지한다. 커밋·push는 하지 않았다.
