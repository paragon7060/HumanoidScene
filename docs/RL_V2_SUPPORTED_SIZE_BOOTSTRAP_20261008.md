# 실패한 크기별 접근 위치에서 시작하는 새 SAC 학습

중간 좌우 small/medium과 상단 좌우 small의 여섯 조합을 모두 학습한다.
기존 크기별 준비 과정은 각 조합에 성공 또는 안전한 접근 후보가 있어야 했다.
새 TRAIN 재확인에서 중간 왼쪽 medium의 선택 후보는 4회 모두 랙 충돌이었다.
이 조합을 제외하거나 성공으로 표시하지 않고, 실제 실패 상태에서 새 경험을
수집할 수 있는 명시적인 초기화 경로를 추가했다.

## 유지하는 조건

- 박스·base 시작 위치·배경·움직이는 firm flap의 기존 randomization을 유지한다.
- 양손 opposing flap 파지, 실제 pad 접촉, 0.25초 유지, 8mm 지지면 clearance를
  성공 조건으로 유지한다. 랙 10N·주변 장애물 5N·reset 대비 낙하 10cm 판정도 같다.
- 후보 발견과 새 TRAIN 재확인의 전체 128회 결과 및 실패 분모를 보존한다.
  중간 왼쪽 medium은 성공 0·unsafe 4로 남고 `measured_success=false`다.
- 후보 측정 행·평가 행·VR 시연을 새 Q 또는 reward 은행에 넣지 않는다.
  새 정책·Q·replay 업데이트는 실제 새 TRAIN 상호작용으로 수행한다.
- base 접근·정지 제어기를 사용한다. 파지 단계의 19개 몸통·팔 목표와 두 jaw는
  SAC가 제어한다. 기존 작은 박스 정책에서 얻은 actor-only TRAIN 참고 경로는
  원래 관측·held x/y/yaw·명령·실패/성공 라벨을 변경하지 않고 보존한다.

## 두 번의 명시적 선택

기본 `TRAIN_measured_supported_size_workplaces_v1` 검사는 계속 엄격하다.
실패 후보 초기화는 별도 `TRAIN_failed_supported_size_workplace_bootstrap_v1`
형식을 사용하며, 아래 두 옵션을 모두 선택해야 한다. 실제 크기·논리/물리
asset pool·위치·출처 hash·disjoint seed와 여섯 조합 검사를 생략하지 않는다.
수치 오류 후보나 새 재확인에서 유효한 시도가 없는 후보는 거부한다.

```bash
python scripts/rl/prepare_size_workplaces.py \
  --discovery-run /absolute/path/to/closed-discovery-run \
  --confirmation-run /absolute/path/to/closed-confirmation-run \
  --selections-json /absolute/path/to/all-six-selections.json \
  --output /absolute/path/to/new-waypoints.json \
  --bootstrap-failed-attempts

python scripts/rl/prepare_size_workplace_actor.py \
  --initial-checkpoint /absolute/path/to/pristine-regional/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/original-physical-contract.json \
  --waypoints /absolute/path/to/new-waypoints.json \
  --output-dir /absolute/path/to/unique-initialization-directory \
  --allow-failed-bootstrap
```

실제 CLI 인자는 각 스크립트의 `--help`로 확인한다. 첫 단계는 정상 종료한
동일 원본 정책·reset 계약의 frozen TRAIN 발견 및 독립 재확인만 받는다.
두 번째 단계는 actor/Q/온라인 replay/성공/누적 보상 은행 0, 사용하지 않은
optimizer를 가진 pristine 정책만 받는다. 기존 learned Q의 재개 경로가 아니다.

## 복원 오류 수정과 검증

전체 trainer 복원에서 nested actual-flap anchor가 새 크기별 목표를 원래
regional 출처와 직접 비교해 거부하는 오류를 발견했다. 검증한 size 계약의
`source_region_workplaces`로 actual anchor의 출처만 비교하고, nested nominal
anchor는 기존 `source_shelf_templates` 검사를 유지하도록 연결했다. actor
가중치·정규화·optimizer·anchor·원래 actor-only 기억은 변경하지 않는다.
새 정책과 replay의 전체 goal 계약은 size 계약을 유지하므로 old Q와 섞이지 않는다.

관련 테스트 86개 통과, GPU integration 1개는 별도 실제 실행으로 확인한다.
2026-10-08 08:38 KST 준비에서 실제 pristine checkpoint를 전체 trainer로
복원하고 128개 첫 TRAIN 상태 모두의 구역·크기 stage를 검증했다. 모델·
normalizer·네 optimizer·anchor·actor-only 기억 12,476행이 원본과 같고,
actor/Q/온라인 replay/성공/누적 보상 은행은 0이었다.

새 seed의 TRAIN 3,072조건, 384조건마다 동일한 전체 DEV128 반복, 독립 FINAL128을
준비했다. 모든 wave는 middle small/medium 각 16개와 upper small 각 32개를
포함한다. FINAL은 준비만 했으며 사용하지 않았다. 준비·복원 검사는 파지
성공이나 학습 개선의 증거가 아니며, 실제 첫 DEV와 학습 후 평가가 필요하다.

기존 Drive 인증은 원래 checkout의 wrapper·supervisor를 통해 재사용한다.
다른 worktree에 인증을 복사하지 않는다. 300초 업로드·크기/MD5 검증·각 형식의
최신 2개 보존을 유지하며 검증 전 파일은 정리하지 않는다. Checkpoint/log
전용 백업에서는 replay/HDF/영상이 로컬에 남는다.
