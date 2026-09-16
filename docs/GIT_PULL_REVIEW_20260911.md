# 2026-09-11 pull 및 검토

`main`을 `bafcc0f`에서 `d22dec5`로 fast-forward했다. 원격 커밋 4개를 반영했으며
검토 시점 HEAD와 origin/main은 같다. commit/push 또는 학습 재시작은 하지 않았다.

## 해결한 통합 문제

- pull 전 수정·미추적 파일 43개와 Git diff를
  `/tmp/humanoid_before_pull_20260911_165822_j1hr6qyj`에 백업했다.
  autostash `e265c5f`도 복구용으로 유지한다.
- 6개 충돌 파일을 해결했다. 새 후보 flap/TCP/접촉 유지/접근 보상 설계를 기준으로
  병합했다. 기존 Drive 저장·체크섬 검증·SAC 데이터 수집 수정은 유지했다.
- 자동 병합된 관측과 reset 계약에 남아 있던 구형 `ready_to_close`, `pad_geometry`
  참조를 제거했다. 새 구현에는 해당 속성이 없어 그대로 실행하면 AttributeError가 난다.
  관측과 계약은 원격 revision 3 정의를 따른다.
- 구형 convex 접촉면 중점으로 새 TCP를 덮어쓰는 경로는 새 보정 TCP 구현으로 대체했다.
  로컬 geometry 도구와 선택적 alignment/closing 함수는 보존했다. 이 두 보상은
  현재 기본 RewardManager에 등록되지 않으며, closing 진단값은 새 최근접 후보에서 구한다.
- `--grasp-debug-vis`를 새 보정 손가락점/TCP/목표/움직이는 중점 표시로 연결했다.
- RL extras에 새 Pillow와 기존 SciPy를 함께 유지했다.
- 파지 후보 테스트의 TCP fixture를 갱신하고 보정점/비보정점 두 경우를 검사했다.
  아키텍처 검사는 VR 진단 도구의 teleop 사용을 허용하되 학습 모듈의 진단 도구
  의존성은 금지하도록 범위를 명시했다.
- RL_FLAP_PICK 문서의 오래된 접근 보상식·TCP·마찰값·성공 조건 설명을 정정했다.

## 학습 전에 검토할 동작 변경

### 충돌 제한이 꺼져 있음 — 기존 요구와 불일치

`configs/rl_pick_arms_only.py`의 `collision_constraints_enabled=False`로 인해
로봇과 랙/주변 물체 충돌이 종료 또는 collision 보상에 반영되지 않는다.
물리 충돌과 센서는 유지된다. 임계값도 0.1N에서 20N으로 바뀌었으므로 기존 요구를
복원하려면 해당 flag와 `obstacle_contact_force`를 모두 변경해야 한다.
패키지 내 동일 preset 및 TaskSpec 기본값도 확인해야 한다.
이번 검토에서는 원격의 의도적인 설정을 그대로 두고 차이를 기록했다.

### 상승 없는 지속 보상 가능

접근은 `4 * (exp(-12*d_now) - exp(-12*d_prev))`이다. 정지 시 0이고,
파지 획득/유지/상실 전환에서는 누적하지 않는다. 단, 별도 orientation은
미파지 상태에서 10cm 이내 정렬만으로 양수가 된다. contact와 stable_grasp도
각각 지속 양수다. 안정된 유효 파지를 높이 변화 없이 유지하면 이 두 항만으로
초당 최대 `3*1.25 + 2*1 = 5.75`를 얻는다(다른 보상/비용 제외).
따라서 이전의 모든 준비 항을 0 이하로 두던 로컬 설계와 다르다.
버티기가 실제 학습에서 우세해지는지는 새 실험으로 확인해야 한다.

### 제어·파지·성공 계약 변경

- `active_arm="right"`: action 8개(오른팔 7 + 그리퍼 1). 왼팔은 고정되어
  policy가 반대손을 움직여 받치는 행동을 학습할 수 없다. `both`는 양팔 제어 옵션이다.
- `flap_contact_region="surface"`: flap 상단 띠만이 아니라 두 후보 판 전체 표면을 허용한다.
- pick 성공의 속도·잔여 손가락 힘 제한이 제거됐다. 초기 안착도 안정성 확인 대신
  고정 0.5초 대기로 바뀌었다. 접촉 유지 판정은 실제 파지 획득 후 최대 0.1초의
  제한된 접촉 누락을 허용한다.
- revision 3 관측/action/파지 계약은 종전 223/227차원·16 action checkpoint와 호환되지 않는다.
  기존 manifest 검증을 우회해 재개하면 안 된다.
- 기존 PPO 관측 정규화 및 탐색 표준편차 문제를 해결하는 변경은 이번 pull에 없다.
  과거 진단의 특정 관측 인덱스는 새 계약에 그대로 적용할 수 없다.

## 검증

- 변경·로컬 Python 파일 81개 AST, shell 문법, TOML 파싱 및 `git diff --check` 통과.
- 병합 관련 핵심 테스트 56개 통과.
- 전체 CPU 테스트: **532 passed, 1 failed, 1 skipped**.
  `CUDA_VISIBLE_DEVICES=''`로 GPU를 비활성화했다. Isaac Sim 물리 실행은 하지 않았다.
- 실패: `tests/test_self_collision.py::test_distance_gradient_agrees_with_fcl`.
  수치 미분 0.06763215 대 해석값 0.02919251로 허용오차 0.002를 벗어난다.
  pull 이전 `bafcc0f`의 `urdf_arm_ik.py`와 `self_collision.py`를 별도 테스트
  프로세스에 로드해 비교해도 동일 값으로 실패했다(15 passed, 1 failed).
  따라서 이번 변경으로 새로 생긴 실패라는 근거는 없다. 원인은 미해결이다.
- skip: ffmpeg가 없는 CPU 테스트 환경의 비디오 인코딩 검사.
- CPU 환경은 robocasa Python 3.11과 `/tmp`에 격리한 usd-core 26.8,
  websockets 17.1, trimesh 5.1.0, python-fcl 0.7.0.8을 사용했다.
  공유 Conda 설치는 변경하지 않았다.
- 전체 로그: `/tmp/humanoid_pull_review_pytest.log`, 비교 로그:
  `/tmp/humanoid_pull_review_baseline.log`, `/tmp/humanoid_pull_review_fcl.log`.
