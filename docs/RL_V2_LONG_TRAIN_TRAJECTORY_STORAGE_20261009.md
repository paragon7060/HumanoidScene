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
