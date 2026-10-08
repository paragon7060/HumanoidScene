# 실제 SAC 정책과 초기화 출처의 metadata 구분

여섯 배치 SAC의 실제 시작 검사에서 `agent.yaml`·progress의 정책은 준비본과
일치했지만 `manifest.json.goal_contract`는 과거 nominal 초기화의 480차원
정책으로 남아 있었다. 현재 정책은 518차원 actor·578차원 critic의 regional
SAC이며, 여섯 구역·크기를 포함한다. Manifest의 물리·보상·성공·안전·DR
필드는 실제 정책과 모두 같았다. 잘못된 설명 metadata와 실제 제어 오류를
구분한다. 현재 실행을 중단하거나 policy/replay를 바꾸지 않았다.

새 실행의 manifest를 처음 만들 때 실제로 읽은 checkpoint의 goal 계약과
actor/Q 업데이트 수를 기록하도록 수정했다. 과거 nominal goal은
`initialization_source_goal_contract`로 보존한다. 선택된 runner·checkpoint·goal
이름과 카운터·JSON 유한성을 확인하며, 실제 trainer 복원 검사는 그대로다.
`agent.yaml`은 복원한 runtime 정책 계약의 기준이다.
기존 literal physical-body SAC는 `physical_body_contract` 이름을 유지한다.
Checkpoint 종류에 맞는 단 하나의 정책 계약을 선택하므로 해당 경로도 보존한다.

실제 정책의 업데이트가 있으면 오래된 입력의 `initialized_not_trained=true`
표시도 그대로 물려받지 않는다. 모델·정규화·optimizer·물리·reward·성공·충돌
판정·randomization·replay에는 영향을 주지 않는다. 최초 업로드 전에 manifest
생성에 반영하므로, 이미 업로드한 파일을 뒤늦게 바꿔 immutable 백업을 깨는
방식이 아니다.

기존 실행의 manifest는 덮어쓰지 않았다. 새 all-six 실행에는 실제 파일과
현재 정책의 출처를 구분한 `verification.json`을 따로 만들었다. 이 수정은
앞으로 시작하는 실행에 적용한다. 기록 정확성 수정이 파지 성공이나 학습
개선을 입증하지는 않는다.
