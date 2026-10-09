# SAC 장기 학습의 기록량 줄이기

기존 장기8개는 TRAIN6,144조건 계획을 계속하며 현재 실행 파일이나 데이터는
바꾸지 않는다. 새 장기를 더 연결할 때 실패 TRAIN의 원시 HDF와 SAC replay를
둘 다 크게 저장하면 공간이 부족할 수 있어 **새 실행의 HDF 기록에만 적용하는
선택 옵션**을 추가했다. 기본은 종전처럼 전부 기록하는 `all`이다.

```bash
# Add to a NEW matching train_batched_staged_goal.py invocation.
--train-trajectory-storage successful-train
```

이 모드는 모든 원래 요청·초기 무효·성공·실패와 실제 transition 수를 metrics에
남기고, 성공과 실패의 실제 전이는 종전처럼 bounded SAC replay에 넣는다.
완료된 실제 안전 양손 pinch·hold·proof lift 성공만 TRAIN HDF로 보존하고,
DEV/FINAL의 모든 실제 HDF 경로와 선택 평가 영상은 그대로 기록한다.
Actor/critic·reward·randomization·안전·success bank와 return bank는 바꾸지 않는다.

`manifest.json`과 HDF manifest의 `executed_trajectory_storage`, 각 outcome의
`trajectory_storage.recorded_HDF_rows`로 원래 수집과 HDF 보관량을 구분한다.
`executed_transition_rows`는 HDF에 쓰지 않은 실패도 포함한 실제 원래 수집량이다.
Replay는 종전과 같이 크기가 제한되므로 오래된 실패 경로의 완전한 원시
이력을 보관하는 모드는 아니다. 전체 TRAIN HDF가 필요한 단계 재현 분석에는
사용하지 않는다. 모델/정규화 checksum과 원래 전체 DEV128 성능 검사는 유지한다.

실제 HDF 쓰기에서 성공만 선택하고 DEV/FINAL 실패를 모두 보존하며, 입력
rows/outcomes를 변경하지 않는 검사를 포함해 recorder 관련21개 검사가 통과했다.
실제 물리 pilot의 새 성능을 검증한 결과는 아니다. 아직 이 옵션으로 시작한
GPU writer는 없고 현재9개 writer는 종전 기록 설정으로 계속한다.

이는 앞으로 쓸 복사본을 줄이는 기능이다. 기존 raw 파일 전송/삭제와 인증
생성은 수행하지 않으며 checkpoint/log 전용 Drive 범위도 그대로다.
5분 업로드·크기/MD5 검증 후 오래된 checkpoint 정리·최신2개와 최종 로그 검증은
[기존 Drive 규칙](RL_GOOGLE_DRIVE.md)을 사용한다.

## 더 긴 원래 무작위화 계획

공유 `prepare_long_region_training.py`는 원래 DEV의 위치·크기 분포를 확인하고
동일한 분포의 새 TRAIN 시작 조건을 생성한다. 중형이 있는 기준은 중간
양쪽small/medium·상단 양쪽small 여섯 조합이 균형을 이뤄야 한다. 기존 기본
설정이 중형을 누락할 수 있었던 부분을 보강했으며, 과거 실행을 소급 변경하지
않는다. 현재 장기들의 실제 전체 평가에서는 여섯 조합이 보존됨을 별도로
검증한 상태다.

손별 닫기v9 후속 후보의 TRAIN6,144조건·DEV17회 계획을 실제 생성했다.
매 wave128요청이고 원래 DEV는 매 TRAIN384 뒤 반복한다. Original box/base/
flap randomization과 독립 FINAL 분리는 유지한다. 이 계획만으로 새 학습
실행·새 성공을 주장하지 않는다. 새 실행은 원래 pilot 종료·최종 Drive 검증·
실제 동일 checkpoint/replay 복원·GPU/저장 여유를 확인해야 한다.
실제 크기 분포·seed 분리·unbalanced 입력 거부를 포함한 관련4검사가 통과했다.

**15:21 실제 대기 등록:** CPU PID363,942, unit
`humanoid-rl-independent-hand-close-long-gpu3-queue-20261009-1520.service`가
현재v9 pilot의 실제 시작을 기다리고 있다. 이후 원래 소유 writer/supervisor의
정상 종료와 최종 checkpoint/log Drive 검증, 실제 whole DEV128 전후 모델 일치,
유한한 동일 모델·실제 held replay와 두 TRAIN bank를 확인한다. GPU3 여유
6,500MiB·일반 디스크20GiB·RAM 저장소35GiB 이상일 때만 새 고유 폴더를 만든다.
저장량은 성공률/경로 길이에 따라 달라져 이 기준이 장기 전체 기록량을
보장하지 않는다. 기존 raw 정리는 승인 없이 수행하지 않는다.

source는 commit `92f78cc51c03723ded8df7cfd9fc95d04389b28c`에 고정했고
778개 코드/기하 SHA를 기록했다. 원래v9의 모든 runtime 모듈과 물리 assets는
같으며 HDF 기록 entrypoint와 장기 계획 생성기만 달라졌다. 문서용 미디어와
인증은 복사하지 않았다. 현재 사용자 renderer overlay는 같은 checksum으로
보존했다. Actor·Q·Adam·normalizer·bounded replay·실제 TRAIN bank를 원래
matching continuation 경로로 이어받고 새 Q 초기화로 바꾸지 않는다.

전체 DEV17회의 정확한 경계 모델 보존·전후 비교용 CPU observer2개도
준비했다. 새 GPU writer가 관측된 뒤에만 등록한다. 현재 대기열과 준비
파일은 새 GPU 실행이나 성공의 증거가 아니다. 전체 TRAIN HDF를 요구하는
기존 접촉 단계 분석은 이 줄인 기록 모드에 연결하지 않는다.
[실제 PID·CUDA·단계](assets/rl_v2_independent_hand_close_long_actual_CPU_queue_20261009.json).
