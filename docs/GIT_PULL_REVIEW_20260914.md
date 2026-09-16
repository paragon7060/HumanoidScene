# 2026-09-14 pull 검토

`main`을 `f93e9e9`에서 `e92b3d9`로 fast-forward했다. 원격 main과
커밋 차이는 0이다. 기존 미커밋 수정 및 untracked 파일은 보존했다.
README 문서 링크 충돌 1건은 4박스 실험 링크와 Drive 링크 모두 유지해 해결했다.
자동으로 stage된 로컬 수정은 다시 unstaged 상태로 돌렸다. Commit/push는 하지 않았다.

복구용 파일 백업은 `/tmp/humanoid_pull_20260914_9z0kozf2`에 있다.
Autostash `5ee61d2`도 삭제하지 않았다. 기존 파일과 비교해 원격에서 변경한
README, docs/README, runners/common 이외의 로컬 파일 내용 변경은 없었다.

## 원격 커밋

- `9e1d80c`: 독립 `rl/multi_box` 모듈에 4박스 PPO 단계별/전체 직접 학습 추가.
  Pick → extract → carry → place 학습과 성공 reset bank 전달, 전체 평가 제공.
  25개 action으로 base/torso/양팔/head/gripper 제어. 기존 8-action SAC checkpoint는
  이 환경에 사용할 수 없고 기존 SAC 진입점이 이 학습으로 바뀐 것도 아니다.
- `ab53802`: Quest RL 보상 디버그와 패널 조절 문서 보강.
- `c523a06`: 선택형 랙 롤러 asset 및 scene/teleop/RL CLI 연결.
  회전 가능한 실린더와 물리 접촉을 사용하며 기본 비활성화다.
  `--rack-rollers` 또는 `KUAVO_RACK_ROLLERS` 설정을 확인해야 한다.
  켜면 박스 지지/이동 물리가 달라지므로 기존 policy의 재생 조건도 달라진다.
- `e92b3d9`: Quest 오른쪽 squeeze 1.5초 유지로 에피소드를 성공 저장하는 조작,
  저장 상태 HUD, RL 보상 디버그의 absolute controller mapping 연결.
  `--rl-reward-debug [0|1]`에서 0/값 생략은 기존 팔 잠금 유지, 1은 디버그에서
  양팔 잠금을 해제한다. 오른쪽 squeeze 성공 표시는 수집기 수동 성공 조작이며
  SAC 환경의 실제 파지 성공 판정을 변경하는 항목이 아니다.

## 기존 SAC에 대한 영향

원격 4개 커밋은 기존 SAC 알고리즘, 한 박스 rewards manager 및 task 설정을
직접 변경하지 않았다. 공통 bootstrap은 랙 롤러 CLI와 debug-only active_arm
override를 추가했다. 기본 롤러 비활성화·override 없음에서는 기존 경로를 유지한다.
학습에 사용한 소스 스냅샷은 pull 대상이 아니므로 그대로다.

실행 중인 학습/다른 사용자 프로세스를 정지하거나 새 학습을 시작하지 않았다.

## 확인

- 변경된 원격 Python 파일 31개 문법 확인 통과.
- 충돌 항목 없음, `git diff --check` 통과.
- GPU를 가린 CPU 테스트 43개 통과: multi_box, SAC/DPPO 기본 연결,
  settling 데이터 수집, Quest runtime 및 collector setup 관련 테스트.
- Isaac 물리/GUI 및 새 4박스·롤러 성능 검증은 수행하지 않았다.
