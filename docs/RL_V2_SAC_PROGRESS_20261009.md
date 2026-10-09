# SAC 양손 파지 진행 — 2026-10-09

목표는 박스와 base 시작 상태가 달라도 안전하게 양손 flap을 집고 유지하며
짧게 드는 **학습된 정책**이다. 현재 목표는 미달성이고 중형 박스 성공도0이다.
원래 위치·크기 여섯 조합의 전체128개 조건으로 비교한다.

**11:54 KST 다음 실제 비교 연결:** 성공/실패 비중 조절 비교의 CPU 대기
PID1,400,160을 확인했다. 원래 v7→팔 전체 랙 회피v8→비중 조절 비교 순서로
각 실행의 정상 종료·최종 Drive 검증과 GPU3/저장소 여유를 확인한다.
동일 actor·관측·원래 전체 배치를 보존한 새 초기 모델을 만들었고,
Q·replay·optimizer는 새로 시작한다. source770개를 commit89fa6d1에 고정했다.
CPU 대기만 실행 중이며 이 비교의 GPU writer는 아직 없다. 기존9개는 계속하고
v7은 11:53 TRAIN2에서 actor89/Q2,402로 실제 actor 갱신도 시작했다.
[실제 CPU 대기·소스·초기 모델·실행 전제](assets/rl_v2_whole_arm_balanced_return_actual_CPU_queue_20261009.json).

**11:45 KST 학습 비중 후보 준비:** 종료된 v6의 실제 TRAIN만 사용해 TD와
return 손실의 갱신 방향을 비교했다. 원래 sampling16batch는 두 gradient가
모두 반대 방향이어서 가중치 증가는 보류했다. 성공/실패 비중을 조절하는
옵션은 두 CPU 표본에서 성공 경로의 Q-return 오차를 더 일관되게 줄였다.
관련 test39개와 새 초기 모델의 실제 저장·학습 재개·Q 복원을 통과했다.
새 물리 writer나 성공률 개선은 아직 아니며 기존9개 학습과 v8 대기는 유지했다.
[실제 값·변경·한계](RL_V2_CRITIC_DIAGNOSTICS_20261009.md).

**11:38 KST v7 첫 실제 TRAIN128 완료:** 성공10·안전 위반53·시간 초과65회,
초기 무효0회였다. Actor0/Q1,590인 critic warmup이므로 학습 후 greedy 성능이
아니다. 원래 탐색 배정33회는 성공0·안전 위반23·시간 초과10회, 나머지95회는
성공10회였다. 안내 경로의 lift 시도도0회였다. 안전 종료53회 중 rack48·box drop3·
lift limit2회였다. 원래 DR·성공·안전 기준을 유지하며 TRAIN2를 이어가고,
뒤이은 전체 DEV128로 학습 성능을 판단한다.
[첫 TRAIN의 실제 원래 조건·mode·종료 근거](assets/rl_v2_URDF_predictive_contact_first_TRAIN128_20261009.json).

**11:16 KST 새 전체 결과:** 강한 성공 유지 비교는 TRAIN2,304조건 뒤 성공6·
안전 위반55·시간 초과67/128회였다. 직전5회보다1회 늘었지만 자기 초기11회보다
낮고 중형0이다. Actor6,778/Q29,158의 실제 평가 전후 모델·optimizer·원래 전체
128조건을 확인했다. 학습은 계속하고 다음 장기 전체평가7개에도 실제 CPU
관찰자를 연결했다. 준비한 파일의 존재를 실행이나 평가 완료로 해석하지 않는다.
[여섯 번째 강한 유지 전체 결과](assets/rl_v2_URDF_strong_gentle_full_arm_sixth_learned_full_DEV24_20261008.json).

**11:12 KST v7 실제 수집 시작:** 예측 제어 비교도 전체 초기 DEV128을 끝냈다.
성공14·안전 위반44·시간 초과57·초기 무효13회이며 actor/Q0인 기준이다.
초기 실제 모델과 원래 여섯 배치를 확인했고 v5·v6의 초기 결과와 같다.
이후 원래128조건의 첫 TRAIN이 시작됐다. 첫 완료 TRAIN에서 안내20% 경로의
실제 접근·닫힘·양손 파지와 새 수집을 확인하고, 학습384 뒤 보조 없는 전체
평가로 판단한다. 현재 결과를 학습 후 성능으로 보지 않는다.
[전체 초기 결과·같은 모델·원래 조건](assets/rl_v2_URDF_predictive_contact_first_full_DEV_20261009.json).

**11:04 KST 새 전체 결과:** 성공 명령 유지 보강 장기는 TRAIN3,072조건 뒤
전체 성공2·안전 위반42·시간 초과84/128회였다. Actor9,208/Q38,880의 평가
전후 같은 모델·optimizer·원래 여섯 배치를 확인했다. 성공2개는 상단 오른쪽
small이고 다른 다섯 조합은0회다. 초기14회보다 낮고 공통 유효115조건도
14→1회여서, 긴 학습의 개선은 아직 확인되지 않았다. 이후 TRAIN33을 계속한다.
[여덟 번째 전체 평가·같은 모델 근거](assets/rl_v2_URDF_servo_guard_eighth_learned_full_DEV32_20261008.json).

**10:58 KST 실제 원인 확인:** 종료된 v6의 전체 평가에서 실행 명령을 재현한
124경로를 같은 actor695/Q4,828로 분석했다. 성공11경로의 held 시작 실제
return 중앙값4.614에 비해 Q는0.523이었다. 안전 위반54경로의 실제 return은
-4.789, 시간 초과59경로는-1.116으로 성공보다 높지 않았다. 명령 재현 오차
기준을 넘은4경로는 Q 분석에서만 제외했고 원래 전체 성공률11/128은 유지했다.
성공 TRAIN21경로/10,549행이 critic에17.9%로 섞이고 실제 return 목적함수도
연결되어 있음을 확인했다. 성공 자료가 빠진 연결 오류는 아니다. 같은 성공
TRAIN 상태의 확률 action 기대 Q와 greedy Q 차이는 중앙값-0.0054여서,
이 자료에서는 무작위성만으로 큰 return 차이가 설명되지 않았다.
[그림·값·수집 범위·원인 해석의 한계](RL_V2_CRITIC_DIAGNOSTICS_20261009.md).

v5·v6·v7의 초기 guard는 파일 checksum까지 동일하다. 모두 유효115·무효13개로,
v7의 새 회귀가 아니다. 무효8개는 안정된 background 박스의 영역 이탈이고
5개는 초기 정착 실패 후 교체되었다. 교체 후 자세로 원래 원인을 추정하지 않는다.
상단 왼쪽 성공2개 중 Env78은 초기 무효, Env90은 초기 유효 timeout이었다.
공통 유효115조건도14→10으로 전체 개선이 아니다. 원래 guard·분모·DR을 유지한다.
[실제 초기 기록·비교](assets/rl_v2_predictive_initial_guard_comparison_20261009.json).

현재 실제 SAC writer9개는 계속하며 기존 여덟 장기의 TRAIN6,144조건 계획도
유지한다. 11:36 실제 PID·CUDA·메타데이터 확인에서는 v5 DEV8·닫힘 정착 DEV20·
접촉 탐색 DEV24·작은 탐색 DEV28·uniform DEV40을 평가했고, 다른 장기들은
TRAIN을 계속했다. v7도 첫 TRAIN을 끝내고 다음 수집을 이어간다. 고정 소스757개와
기존 Drive를 사용하며9개 관리자 모두 최근 백업 검증과 오류 없음이 확인됐다.
v8은 CPU4,186,466이 앞선 v7의 정상 종료·최종 Drive 검증을 기다린다.
[실제 v7 startup](assets/rl_v2_URDF_predictive_contact_SAC_actual_startup_20261009.json).

**10:31 KST 실제 후속 시작:** 안쪽20mm 파지점 v6의 학습 후 전체128조건은
성공11·안전 위반54·시간 초과63회였다. 초기14회보다 낮고 중형0이지만, 상단
왼쪽small에서 초기0→2회가 기록됐다. Env78·90은 실제 양손 opposing 파지와
0.267초 유지·proof lift·안전을 통과했다. 전체 일반화 개선으로 보지는 않는다.
정상 종료와 최종 Drive 검증을 확인했고, v7 예측 제어의 GPU3 writer38,470·
supervisor38,365와 소스757개·CPU 관찰자5개가 실제로 시작됐다. 아직 첫 runtime
progress와 전체 초기 평가가 없어 학습 성과를 주장하지 않는다. 수정 v5와 기존
일곱 장기는 계속되고, 새 손·팔 랙 회피 v8은 실제 CPU4,186,466이 v7 종료를 기다린다.
[안쪽 파지점 학습 후 전체 결과](assets/rl_v2_URDF_interior_contact_repaired_first_learned_full_DEV4_20261009.json),
[v7 실제 시작·기존 Drive·소스·관찰자 검증](assets/rl_v2_predictive_contact_actual_GPU3_start_20261009.json).

**10:11 KST 새 전체 결과:** 수정 torso 장기는 TRAIN384 뒤 성공10·안전 위반52·
시간 초과66/128회로 끝났고 이어 학습한다. 초기14회보다 낮고 중형0이라 개선으로
보지 않는다. Uniform은 TRAIN3,456 뒤 성공2·안전 위반63·시간 초과63/128회였다.
평가 전후 같은 실제 모델·유한 optimizer/정규화와 원래 전체 조건을 확인했다.
기존 장기 실행은 유지한다.
[수정 torso 첫 학습 후 전체 결과](assets/rl_v2_URDF_upright_contact_repaired_first_learned_full_DEV4_20261009.json),
[uniform 아홉 번째 전체 결과](assets/rl_v2_URDF_regional_goal_ninth_learned_full_DEV36_20261008.json).

손바닥·팔뚝과 고정 랙 형상19개를 고려하는 선택형v8 탐색을 구현했다. 실제 목표
80mrad/URDF 제한을 통과한 pending 자세에서 형상을 계산하며 기존20% TRAIN에만
적용한다. 보조 없는 평가·SAC 분포·닫힘/충돌/성공/무작위화 기준은 유지한다.
30검사와 실제384상태의 초기21목표 일치·저장/재개/영상 복원을 확인했다.
진단18상태의 가정한 팔 목표17개에서 명목 간격이 늘었지만 실제 rollout이나
새 파지 성공은 아직 아니다. GPU3 순차 대기와 실제 시작은 별도 PID로 확인한다.
[변경·방법·수치·검증 한계](RL_V2_WHOLE_ARM_CLEARANCE_SAC_20261009.md).

10:22까지 추가로 끝난 전체평가에서도 큰 탐색은5→8회, 닫힘 정착은7→8회로
조금 늘었지만 둘 다 초기11회보다 낮고 중형0회였다. 작은 탐색6회·접촉 탐색7회도
유지됐다. 모든 결과는 같은 실제 평가 모델과 원래 전체128조건으로 확인했다.
기존 최고13회 모델을 보존하며 단기 변화만으로 일반화 개선을 주장하지 않는다.
[큰 탐색 DEV28](assets/rl_v2_URDF_full_arm_seventh_learned_full_DEV28_20261008.json),
[닫힘 정착 DEV16](assets/rl_v2_URDF_settled_contact_fourth_learned_full_DEV16_20261009.json),
[작은 탐색 DEV24](assets/rl_v2_URDF_gentle_full_arm_sixth_learned_full_DEV24_20261008.json),
[접촉 탐색 DEV20](assets/rl_v2_URDF_perceived_contact_fifth_learned_full_DEV20_20261008.json).

v8의 실제 CPU 대기 PID4,186,466은 v7 정상 종료·최종 Drive와 GPU3 여유를
기다린다. v8 GPU 학습은 아직 시작 전이다. 새 payload는 기존 우리 RAM 경로에
쓰며 일반 디스크와 RAM의 남은 저장량을 나눠 관리한다. 기존 checkpoint/log
백업과 진행 중인 장기는 유지한다.
[실제 대기·소스767개·기존 writer9개 확인](assets/rl_v2_whole_arm_clearance_actual_GPU3_queue_verified_20261009.json).

**09:06 KST의 진행:** 원래 무작위화를 유지한 SAC writer9개와 기존 Drive
검증을 재확인했다. 수정 GPU3 장기는 첫 TRAIN128을 성공10·안전 위반46·
시간 초과72회로 마쳤다. Actor0·Q1,590인 첫 수집이며 학습 후 성능은 아니다.
이후 세 번째 TRAIN step421·actor477·Q3,956까지 계속됐다. TRAIN6,144·
replay40만·65배치 장기 실행을 유지한다. 새v6도 초기 전체14/128과 첫
TRAIN128을 마쳐 성공12·랙 안전 종료51·시간 초과65회였다. 안내33회는
성공0회, 닫힘 명령0행이었다. Actor0인 수집이며 학습 후 성능은 아니다.
이후 두 번째 TRAIN step571·actor149·Q2,642로 이어졌다. 두 새 실행의
학습 후 전체 결과는 아직 없다.
[새 장기의 전체 초기 결과·같은 실제 모델](assets/rl_v2_URDF_upright_contact_repaired_first_full_DEV_20261009.json).

닫힘 중 flap 이동을3틱 앞서 예측하되8mm로 제한하는 v7을 구현하고 관련
22검사·실제384상태의 초기 목표 일치·저장/재개/영상 복원을 확인했다. v6에서
닫힘 경험이 없었으므로 실제 닫힘까지 갔던 v5의 원래5mm 지점을 유지한다.
고정 소스757개를 사용하는 CPU 대기열3023114가 v6 정상 종료·최종 Drive
검증을 기다린다. **v7 GPU 학습은 아직 시작 전**이며 기존9개 writer는 유지한다.
[변경·확인 범위](RL_V2_PREDICTIVE_CONTACT_SAC_20261009.md),
[실제 대기열·9개 학습](assets/rl_v2_predictive_contact_actual_GPU3_queue_verified_20261009.json).

09:07 KST에 강한 성공 유지 비교의 다섯 번째 전체평가도 완료됐다.
TRAIN1,920 뒤 성공5·안전 위반56·시간 초과67/128회이며 중형0이다.
Actor5,564·Q24,304의 평가 전후 실제 checkpoint SHA256이 같았다.
초기11회보다 낮고 공통 유효115조건도11→4회라 개선으로 판단하지 않는다.
기존 GPU1 writer2292102는 이어 학습하며 다음 DEV24를 CPU3111246이
관찰한다. 실제 GPU 학습을 종료하거나 평가 자료를 학습에 넣지 않는다.
[강한 성공 유지 최신 전체 결과](assets/rl_v2_URDF_strong_gentle_full_arm_fifth_learned_full_DEV20_20261008.json).

완료된 첫 TRAIN 복사본80,977행·SHA256과 실제 종료128개를 대조했다.
접근 안내33회 중26회는 표면 접근 단계,11회는 닫힘까지 갔지만 실제 양손
파지/성공은0회였다. 다시 열린19구간에서 닫힘 중앙값은 좌36.0%·우42.9%,
flap–TCP 이동 차이 중앙값9.18mm였다. 랙 충돌44회는 오른쪽 그리퍼 몸체22·
왼팔4번 링크12회에 집중됐다. 닫힘 완료까지 움직이는 flap을 추적하는 제어와
그리퍼/팔 전체의 안전 진입을 다음 비교 대상으로 삼는다. 안전·성공 기준을
완화하거나 초기/보조 성공을 학습 개선으로 표시하지 않는다.
[첫 실제 수집](assets/rl_v2_upright_contact_repaired_first_TRAIN_actual_20261009.json),
[측정·그림·분석 한계](RL_V2_UPRIGHT_CONTACT_SAC_20261009.md#첫-완료-train의-실제-실패-원인).

09:34 KST의 전체 손·팔 기하 분석은 랙 실패44회 중43회를 대조했다.
그중33회는 명목상 앞 기둥이 최근접 부위였고,41회는 계산상 간격10mm 미만이었다.
그리퍼 기준점은 마지막 제어 직전256자세 모두 손끝보다 바깥쪽이어서
몸체 뒤집기만을 원인으로 채택하지 않는다. 전체 손·팔의 기둥 회피를 다음
진입 제어 비교로 정했다. 실제 접촉 힘과 명목상 기하/마지막 action 전 자세의
차이를 명시하며 이 계산을 새 성공·새 충돌 판정으로 쓰지 않는다.
[측정·그림·계산 한계·재현](RL_V2_RACK_ENTRY_GEOMETRY_20261009.md).

기존 uniform은 TRAIN3,072 뒤 전체3/128회, 작은 탐색은 TRAIN1,920 뒤
전체6/128회였다. 각각 자기 초기14·11회보다 낮고 중형은 여전히0회다.
08:11에 수정 장기는46,659행·Q778과경계 제안 거부37건에도 계속 실행됐다.
[실제 경계 복구](assets/rl_v2_upright_repaired_actual_boundary_recovery_20261009.json).
새 몸통 제어의 첫 완료 TRAIN에서 접근·닫힘·유지 경험이 늘어나는지 확인한
뒤, 같은 모델의 보조 없는 전체평가로 수정 효과를 판단한다. 초기 성공이나
탐색 보조의 성공을 학습된 일반화 성공으로 바꾸어 표시하지 않는다.
[uniform 전체 결과](assets/rl_v2_URDF_regional_goal_eighth_learned_full_DEV32_20261008.json),
[작은 탐색 전체 결과](assets/rl_v2_URDF_gentle_full_arm_fifth_learned_full_DEV20_20261008.json).

장기 저장량에 평가용 모델 복사본을 포함해 다시 계산했다. 불변 모델38개
약3.46GiB를 RAM에서 로컬 디스크로 옮겨 원래 경로·SHA256을 보존했다.
학습 checkpoint·replay·HDF·로그는 건드리지 않았다. 종료 raw4개 약29.8GiB의
추가 Drive 업로드·로컬 정리는 자동 승인 검토가 별도 승인을 요구하여
시작하지 않았다. 기존 학습·checkpoint 백업은 유지한다.
[실제 공간·이동·보존 조건](RL_V2_LONG_RUN_STORAGE_20261009.md).
[문제·변경·설정·검증 범위](RL_V2_UPRIGHT_CONTACT_SAC_20261009.md).

종료된 오류 실행의44,020개 실제 전이에서 몸통 목표와 실제 관절 자세를
대조했다. Pitch 목표는약1.214°로 유지됐고 추종 오차 중앙값은0.214°였지만
오류 상태에서는6.644°·X 약40.6mm의차이가 있었다. 범위 밖 목표를 명령한
문제는 아니며, 실제 물리 추종 이탈을 한 틱 IK가 회복하지 못한 제안을
전체 배치 오류로 처리했던 문제다. 실제 torque 포화/중력보상 부족까지는
이 자료로 판정할 수 없다. 추가 제어기 변경 대신 수정 장기 실행의 완료
TRAIN과 보조 없는 전체평가를 확인한다.
[실제 목표·측정 오차·그림](RL_V2_UPRIGHT_CONTACT_SAC_20261009.md#실제-몸통-목표와-물리-자세의-차이).

07:30 실제 확인에서 수정GPU3는초기평가step241, v4는TRAIN384뒤최종평가
step781로 진행했다. 기존일곱장기비교도유지하며Drive검증오류는없었다.
GPU1 강한성공유지의다음DEV20/전체128결과를확인하는CPU관찰PID1,595,216도
실제등록했다. 준비파일을실행근거로쓰거나부분평가를완료성공률로쓰지않는다.

이후 v4의 전체 평가 128회는 성공12·안전 위반51·시간 초과65회로 끝났다.
자기 초기11→12이며 중간 왼쪽small9·상단 오른쪽small3·다른 네 조합0회다.
공통 유효115조건도11→12·기존 성공 상실3·새 성공4회여서 안정적인 일반화
개선으로 판단하지 않는다. Actor695/Q4,828의 실제 평가 전후 같은 모델과 유한 tensor를
확인했고 정상 종료·최종 Drive 검증을 마쳤다.
[전체 평가·같은 모델](assets/rl_v2_URDF_motion_feedback_first_learned_full_DEV4_20261009.json).
07:34 KST에 새 v6 writer1,624,488·CUDA3·수정 소스3740689/749개를 확인했다.
원래 TRAIN384·초기/학습 후 전체 DEV각128·20mm 접선 지점을 쓰며 현재 초기화 중이다.
새 파지/학습 개선은 아직 확인 전이다. CPU 평가 관찰의 초기 progress 미생성 대기 오류도
고쳐 실제 새 관찰 PID로 교체했으며 GPU 학습은 중단하지 않았다.
[실제 후속 시작과 관찰](assets/rl_v2_URDF_interior_contact_repaired_actual_launch_20261009.json).

08:20 KST에 안쪽 접촉 지점 v6도 원래 전체 초기128조건을 마쳐 성공14·안전
위반44·시간 초과57·초기 무효13회였다. Actor/Q0인 기준이며 학습 개선이 아니다.
이후 실제 첫 TRAIN을 시작했다. 몸통/팔 제어를 고친 장기는 step841까지
진행했고 완료된128경로의 데이터 분석을 기다린다. 완료된 별도 복사본만 읽어
구역·원래 탐색 선택별 목표/측정 pitch·X/Z와 안전 종료를 비교하는 CPU 분석을
연결했다. 10개 검사와 이전 실제44,020행의32개 필드 재현이 통과했다.
[후속 전체 초기 결과](assets/rl_v2_URDF_interior_contact_repaired_first_full_DEV_20261009.json),
[완료 자료 분석의 실제 연결](assets/rl_v2_completed_TRAIN_torso_analysis_actual_observer_20261009.json).

**추가 수정:** v6의 첫 완료 TRAIN은 성공12/128이지만 안내33회에서 닫힘은0행이었다.
새 v7은 실제 닫힘이 관찰됐던5mm 지점에 연속 측정 이동3틱·최대8mm의 위치
예측을 추가한다. 원래 닫힘·파지·안전 조건과 보조 없는SAC평가는 유지한다.
최종22검사와 실제 완료 상태384개에서 초기 목표21개 동일·저장/재개/Q영상
복원을 확인했다. 새 실제 학습 효과는 아직 미확인이며 기존 v6 종료·Drive
최종 검증 후 별도 비교로 연결한다.
[실제 수집·수정·확인 범위](RL_V2_PREDICTIVE_CONTACT_SAC_20261009.md).

08:55 KST의 성공 명령 유지 보강 전체평가는 TRAIN2,688조건 뒤3/128회였다.
자기 초기14→13→8→6→5→6→8→3이고 중형0이므로 장기 성능 반등은 확인되지
않았다. Actor7,993/Q34,018의 실제 평가 전후 같은 모델을 대조했다.
[전체 평가·같은 모델](assets/rl_v2_URDF_servo_guard_seventh_learned_full_DEV28_20261008.json).

## 최근 전체 평가

| 활성 비교 | 완료한 TRAIN 조건 | 자기 초기 → 학습 후 전체 성공/128 | 판단 |
|---|---:|---|---|
| 기존 uniform 팔 범위 | 3,456 | 14→8→4→1→3→2→1→4→3→2 | 초기보다 낮음 |
| 성공 명령 유지 보강 | 3,072 | 14→13→8→6→5→6→8→3→2 | 반등 유지 실패·중형0 |
| 전체 팔 범위·큰 탐색 | 2,688 | 11→10→7→8→5→5→5→8 | 최근 소폭 증가·초기보다 낮음 |
| 전체 팔 범위·작은 탐색 | 2,304 | 11→11→13→8→6→6→6 | 최고13 모델 보존; 최근 다시 감소 |
| 작은 탐색·강한 성공 유지 | 2,304 | 11→10→10→8→6→5→6 | 최근 소폭 증가·초기보다 낮음 |
| 관측 기반 연속 접촉 탐색 | 1,920 | 11→9→11→13→7→7 | 최고13 모델 보존; 다시 감소·중형0 |
| 측정된 그리퍼 정착 후 들기 | 1,536 | 11→7→7→7→8 | 최근 소폭 증가·초기보다 낮음 |
| 현재 flap 추적·정밀 닫힘 | 384·계획 정상 종료 | 11→8 | 종료 로그/체크포인트 Drive 검증 완료 |
| 닫힘 위치·이동 추종 보강 v4 | 384·계획 정상 종료 | 11→12 | 중형0; 소폭 증가, 안정적 일반화 미확인 |
| 몸통 위치 조절·진입 후 유지 v5 | 384 이후 장기 수집 진행 | 14→10 | 중형0·첫 안내0/33; 장기 계속 |
| 안쪽20mm 접촉 지점 v6 | 384·계획 정상 종료 | 14→11 | 상단 왼쪽2회·중형0·Drive 최종 검증 |
| 원래 지점·닫힘 중 이동 예측 v7 | 전체 초기 완료·첫 TRAIN 진행 | 초기14; 학습 후 전체 평가 대기 | 3틱 예측·최대8mm·기존 판정 유지 |
| 손바닥·팔뚝 랙 회피 v8 | 실제 CPU 대기 | 원래 v7 종료 뒤 GPU3 비교 예정 | 30검사 통과; 물리 수집/새 성공 미확인 |

작은 탐색의 첫 학습 후 결과는 성공11·안전 위반56·시간 초과61·초기 무효0이었다.
중간 왼쪽small7·상단 오른쪽small4, 나머지는0이다. 안전 위반은랙55·낙하1회다.
Actor702/Q4,856의 실제 평가 전후 모델과 optimizer1,014개 유한 tensor를
대조했다. 초기와 공통 유효115조건은11→10회로 안정적인 개선을 주장하지 않는다.
[전체 결과·같은 모델](assets/rl_v2_URDF_gentle_full_arm_first_learned_full_DEV4_20261008.json).

01:57 KST에 작은 탐색의 두 번째 학습 후 전체평가가 완료됐다. TRAIN768조건
뒤 성공13·안전 위반67·시간 초과48·초기 무효0/128이었다. 중간 왼쪽small5·
상단 오른쪽small8, 다른 네 조합은0이다. 초기와 공통 유효115조건은11→12회,
기존 성공 상실5·새 성공6회였다. 성공은 소폭 증가했지만 랙 충돌도66회로
많고 중형은0이므로 안정적인 일반화 개선으로 판단하지 않는다. Actor1,915/
Q9,708의 실제 평가 전후 같은 모델·optimizer1,028개 유한tensor를 대조하고
이 평가에 대응하는 모델을 별도로 보존했다.
[두 번째 전체 결과·같은 모델](assets/rl_v2_URDF_gentle_full_arm_second_learned_full_DEV8_20261008.json).

대표 영상은 위 평가의 **상단 오른쪽small 성공**이다. 초기 base는 횡방향
-1.5cm·밖으로8.2cm·yaw -3.2°가 적용됐고, 접근 후 양손 opposing 파지와
0.267초 유지·실제 들기·안전 조건을 통과했다. 종료 시 랙 지지면 여유는4.48cm다.
같은 요청의 자기 초기 평가는timeout이었지만 flap 추첨·접촉 이력까지 같다고
가정하지 않는다. 탐색 보조 없는 학습된 greedy 정책의 실제 측정 자세를
기록한 영상이며, 여섯 배치 전체의 일반화 성공을 뜻하지 않는다. 화면의
`Q9708`은Q 갱신 횟수이고Q-value가 아니다.

![최근 학습 후 상단 오른쪽small 성공](assets/rl_v2_gentle_DEV8_upper_right_learned_success_20261009.png)

[재생 가능한 H264 영상](assets/rl_v2_gentle_DEV8_upper_right_learned_success_20261009.mp4) ·
[모델·base 변화·실제 성공·전체 재생 검증](assets/rl_v2_gentle_DEV8_upper_right_learned_success_verified_20261009.json).

전체 팔·큰 탐색은actor1,909/Q9,682의 전체128조건에서7회였다.
원래uniform은actor4,349/Q19,444에서3회였다. 각자의 초기 기준과 비교하며
실제 flap 추첨·접촉 이력까지 같은 반복이라고 가정하지 않는다.
[큰 탐색 전체 결과](assets/rl_v2_URDF_full_arm_second_learned_full_DEV8_20261008.json),
[uniform 전체 결과](assets/rl_v2_URDF_regional_goal_fourth_learned_full_DEV16_20261008.json).

01:54 KST의 큰 탐색 세 번째 전체평가는 TRAIN1,152조건 뒤 성공8·안전 위반71·
시간 초과49·초기 무효0/128이었다. Actor3,132/Q14,574의 평가 전후 같은 실제
모델을 확인했으며 자기 초기11회보다 낮다. 중간 왼쪽small4·상단 오른쪽small4,
중형과 다른 구역은0이다.
[세 번째 전체 결과](assets/rl_v2_URDF_full_arm_third_learned_full_DEV12_20261008.json).

강한 성공 유지 비교의 첫 학습 후 전체평가는10·안전 위반48·시간 초과70,
초기 무효0/128이었다. Actor695/Q4,828의 평가 전후 같은 실제 모델·optimizer
1,014개 유한tensor를 대조했다. 성공 명령 유지 보강의 세 번째 전체평가는
6·랙 충돌43·시간 초과79·초기 무효0/128이었다. Actor3,138/Q14,598의 같은
모델·optimizer1,098개 유한tensor를 대조했다. 두 비교 모두 중형은0이다.
[강한 유지 전체결과](assets/rl_v2_URDF_strong_gentle_full_arm_first_learned_full_DEV4_20261008.json),
[세 번째 보강 전체결과](assets/rl_v2_URDF_servo_guard_third_learned_full_DEV12_20261008.json).

01:40 KST의uniform 다섯 번째 전체 평가도성공2·랙 충돌61·시간 초과65,
초기 무효0/128이었다. Actor5,561/Q24,292의같은실제모델과optimizer를
확인했다. 더긴 학습의반등은 아직없다.
[다섯 번째 전체결과](assets/rl_v2_URDF_regional_goal_fifth_learned_full_DEV20_20261008.json).

02:07 KST의 이전 접촉 탐색 첫 학습 후 전체평가는 TRAIN384조건 뒤 성공9·
랙 충돌63·시간 초과56·초기 무효0/128이었다. Actor695/Q4,828의 평가 전후
같은 실제 모델·optimizer1,014개 유한tensor를 확인했다. 자기 초기11→9,
공통 유효115조건도11→9이며 중형은0이다. 탐색의 학습 효과는 확인되지 않았다.
[첫 학습 후 전체 결과](assets/rl_v2_URDF_perceived_contact_first_learned_full_DEV4_20261008.json).

02:56 KST에 강한 성공 유지의 두 번째 전체평가가 완료됐다. TRAIN768조건 뒤
성공10·안전 위반59·시간 초과59·초기 무효0/128이었다. 자기 초기11→10→10이며
공통 유효115조건은11→10이다. 02:39 KST의 성공 명령 유지 보강 네 번째 평가는
TRAIN1,536조건 뒤 성공5·안전 위반53·시간 초과70·초기 무효0/128이었다.
자기 초기14→13→8→6→5이고 공통 유효115조건은14→3이다. 실제 평가 전후
같은 모델과 optimizer 갱신 횟수를 대조했으며 두 비교 모두 중형은0이다.
[강한 유지 두 번째 전체 결과](assets/rl_v2_URDF_strong_gentle_full_arm_second_learned_full_DEV8_20261008.json),
[보강 네 번째 전체 결과](assets/rl_v2_URDF_servo_guard_fourth_learned_full_DEV16_20261008.json).

03:40 KST의 uniform 여섯 번째 전체평가는 TRAIN2,304조건 뒤 성공1·안전
위반46·시간 초과81·초기 무효0/128이었다. Actor6,784/Q29,184의 실제 평가
전후 같은 모델과 optimizer를 대조했다. 자기 초기14회보다 낮고 중형은0이다.
더 긴 수집만으로의 개선은 아직 확인되지 않았다.
[여섯 번째 전체 결과](assets/rl_v2_URDF_regional_goal_sixth_learned_full_DEV24_20261008.json).

03:55 KST의 작은 탐색 세 번째 전체평가는 TRAIN1,152조건 뒤 성공8·안전
위반62·시간 초과58·초기 무효0/128이었다. 자기 초기11→11→13→8이고 공통
유효115조건은11→8이다. 최고13회 모델은 계속 보존하지만 반등이 유지되지
않았다. 03:53 KST의 큰 탐색 네 번째 전체평가는 TRAIN1,536 뒤 성공5·안전
위반73·시간 초과50·초기 무효0/128이었다. Actor/Q는 각각3,139/14,602와
4,351/19,450이며 실제 평가 전후 같은 모델과 optimizer를 대조했다.
두 비교 모두 중형은0이다. 장기 수집과 새 실제 접촉 제어 비교를 계속한다.
[작은 탐색 세 번째 결과](assets/rl_v2_URDF_gentle_full_arm_third_learned_full_DEV12_20261008.json),
[큰 탐색 네 번째 결과](assets/rl_v2_URDF_full_arm_fourth_learned_full_DEV16_20261008.json).

원본 장기 비교는TRAIN3,072를 정상 종료했고 마지막5/128이었다.
최종 checkpoint와 닫힌 로그의Drive 검증까지 끝났지만 목표 달성은 아니다.

04:00 KST에 닫힘 정착 v2의 첫 학습 후 전체평가가 완료됐다. TRAIN384조건
뒤 성공7·안전 위반62·시간 초과59·초기 무효0/128이었다. Actor696/Q4,830의
실제 평가 전후 같은 모델을 대조했으며 자기 초기11회보다 낮다. 기존 버전은
장기 수집을 계속하고, 새 v3의 현재 flap 추적·정밀 닫힘 효과를 별도로 확인한다.
[정착 v2 첫 학습 후 전체 결과](assets/rl_v2_URDF_settled_contact_first_learned_full_DEV4_20261009.json).

04:01 KST의 이전 접촉 탐색 v1 두 번째 전체평가는 TRAIN768조건 뒤 성공11·
안전 위반44·시간 초과73·초기 무효0/128이었다. Actor1,908/Q9,680의 실제
평가 전후 같은 모델을 대조했다. 자기 초기11→9→11로 회복했으나 초기보다
높지는 않고 중형은0이다. 탐색 보조 없는 학습 후 평가다.
[접촉 v1 두 번째 전체 결과](assets/rl_v2_URDF_perceived_contact_second_learned_full_DEV8_20261008.json).

08:03–08:05 KST에 접촉 탐색 v1·정착 v2·큰 탐색의 다음 전체평가도
완료됐다. TRAIN1,536/1,152/2,304조건 뒤 성공은 각각7/7/5회, 안전 위반은
68/83/53회였다. 이전 v1의13회 반등도 유지되지 않았고 모든 중형은0회다.
같은 실제 평가 모델과 optimizer를 대조했으며 독립 FINAL은 아직 쓰지 않았다.
장기 학습은 계속하되 기간 연장만으로 충분하다는 근거는 없다. 수정 v5의
완료 수집에서 실제 양손 접촉 경험·몸통 경계 복구·성공 명령 유지가 늘어나는지
확인한 뒤 후속 변경을 정한다.
[접촉 v1 전체 결과](assets/rl_v2_URDF_perceived_contact_fourth_learned_full_DEV16_20261008.json),
[정착 v2 전체 결과](assets/rl_v2_URDF_settled_contact_third_learned_full_DEV12_20261009.json),
[큰 탐색 전체 결과](assets/rl_v2_URDF_full_arm_sixth_learned_full_DEV24_20261008.json).

## 지금 적용하는 방법

GPU3 여섯 실행과GPU0·GPU1·GPU2 각 한 실행, 총 아홉 SAC writer를 유지한다. GPU2 실행은 기존20% 탐색
episode에 측정 관절·flap 관측으로 랙 앞 진입, 표면 접근/축 정렬, 닫힘 유지,
25mm 들기 시도를 연결한다. 접촉이나 성공 판정을 탐색 입력으로 쓰지 않는다.
기존 jaw gate와 PD 제한, 원래 박스/base/flap 무작위화·보상·성공·안전 기준을
유지한다. 실제 명령/환경 보상으로SAC를 학습하고 전체 평가는 보조 없는
greedy 정책으로 한다. [구현·검사·실제 실행·방법 그림](RL_V2_PERCEIVED_CONTACT_SAC_20261008.md).

장기 일곱 실행은TRAIN6,144·replay200만·65배치를 끝까지 진행하며384 TRAIN조건마다
같은 전체128개 요청을 평가한다. 초기 기준만으로 개선을 주장하지 않고,
TRAIN 탐색의 성공도 학습된 greedy 정책의 일반화 성공과 따로 확인한다.
첫 실제TRAIN128은성공9·안전 위반42·시간 초과77이었다. 접촉 탐색 선택25조건은
성공0회이므로 효과를 주장하지 않는다. 완료된 수집 replay의 실제 손·그리퍼·
접촉 경로를 분석해 수정할 부분을 결정한다. Q와actor 업데이트는 실제 실행에서
진행 중이며 독립FINAL은 아직 사용하지 않았다.

완료된 첫 두 TRAIN 배치159,438행에서 들기16회 모두 시작 전/직후 양손 파지가
없었다. 실제 닫힘 비율 중앙값은좌45.1%·우52.5%였다. 닫힘 명령만으로 너무
빨리 들기 전환을 한 것을 확인해 `full-arm-settled-contact` 비교를 추가했다.
최소24틱 닫힘·측정된 닫힘85% 이상·6틱 정착·현재 접촉 지점6mm를 확인하고
들기를 시도한다. 닫을 때 접촉 목표를랙 좌표에 고정하고 실제 접촉/성공 정보를
탐색 입력으로 쓰지 않는다. 원래 성공 기준은 그대로다.
[완료 수집의 실제 타이밍 분석·그림·수정](RL_V2_PERCEIVED_CONTACT_SAC_20261008.md).

같은 완료 replay를 단계별로 재계산하면 상단 탐색20경로 모두 표면 접근
단계에 들어가지 못했다. 반면 중간 위치27경로는 모두 표면 접근을 시작했다.
랙 앞 준비 단계의 양손 닫힘 명령은0행이고 한 손 닫힘도11행뿐이어서,
접근 전 조기 닫힘을 주된 실패 원인으로 단정하지 않는다. 상단 진입의 거리·
축 정렬을 따로 점검해야 한다. 수정 버전은 탐색 지속 시간을180→360틱으로
늘렸고, 그 효과는 실제 수집·학습 후 평가로 확인한다.
[완료 TRAIN 단계 분석](assets/rl_v2_perceived_contact_completed_TRAIN_phase_analysis_20261009.json).

01:30 KST에새 GPU0 writer353905의 실제 초기DEV 진입·743개 고정 소스·
원래 여섯 조합/6,144 TRAIN/128 DEV/65배치·중력보상18관절·첫Drive 검증을
확인했다. `CUDA_VISIBLE_DEVICES=0`, Kit단일GPU0, 메모리17,570MiB다.
현재일곱 writer가 살아 있고 기존 여섯 비교와 다른 사용자 작업은 유지한다.
02:06 KST에 새 수정의 전체 초기평가가 끝나 성공11·안전 위반48·시간 초과56·
초기 무효13/128이었다. 실제 저장 모델/정규화75개가 준비한 초기 모델과 같고
전체257개tensor가 유한했다. Actor/Q 갱신0인 자기 초기 기준이며 새 학습의
개선을 뜻하지 않는다. 첫TRAIN 수집과Q 갱신을 시작했고 학습 후greedy 결과는
아직 대기 중이다.
[실제 시작·계약·백업](assets/rl_v2_URDF_settled_contact_SAC_actual_startup_20261009.json).
[전체 초기 결과·같은 모델](assets/rl_v2_URDF_settled_contact_first_full_DEV_20261009.json).

02:31 KST에 수정 버전의 첫 실제TRAIN128이 완료됐다. 성공10·안전 위반39·
시간 초과79·초기 무효/수치 실패0회다. Greedy103조건이10회 성공했고 탐색
선택25조건은0회였다. 닫힘 명령1,815행을 실행했지만 들기 전환은0회였다.
너무 이른 들기는 막았으나 실제 파지 성공 증가를 확인하지 못했다. 완료된
TRAIN의 읽기 전용 복사본에서 측정 닫힘·정착·고정/현재 목표 거리 조건을
구분해 다음 수정의 근거를 확인한다. 기존 일곱 학습은 계속하며 원래 기준을
유지한다.
[첫 완료 TRAIN·실제 모델](assets/rl_v2_settled_contact_first_TRAIN_actual_20261009.json).

## 현재 flap 추적 비교의 실제 시작

닫기 정착 v2의 완료 TRAIN을 재계산한 결과, 닫기를 시도한8경로는 이미
측정 닫힘85%·24틱 닫기·6틱 정착을 충족했다. 하지만 양손 현재 접촉점까지
최소 거리는11.77–13.50mm인 반면, 옛 랙 고정 목표에는0.16–2.62mm까지
접근했다. flap이 움직인 뒤 옛 지점에서 빈 파지를 한 것이다. 탐색25조건의
실제 양손 opposing pinch는0회였으며, 상단11조건은표면 접근 단계에도
도달하지 못했다. CPU/CUDA 재계산의 최대 FK float 차이1.86e-8m는
별도 허용오차1e-7m로 기록하며 정수 통계와 실행 명령을 별도로 대조했다.
[실제 닫힘 조건과 실패 근거](assets/rl_v2_settled_contact_completed_TRAIN_gate_analysis_20261009.json).

v3는 선택된 TRAIN20%에서 현재 flap 자세를 추적하고 양손 거리6mm·축 정렬
0.25rad 이내일 때 닫는다. 닫은 뒤 거리가12mm 또는 축 오차0.30rad를
넘으면 열고 재접근하며, 들기 전까지 축 정렬을 계속한다. 기존24틱·85%·6틱
정착·현재점6mm 들기 조건과 원래 무작위화·보상·성공·안전 기준은 유지한다.
관련59검사와 초기 모델 저장·학습 재개·영상 모델 복원이 통과했다.

03:22 KST에 GPU3 writer1,807,053의 실제 초기 평가 진행, `CUDA_VISIBLE_DEVICES=3`,
Kit단일GPU3, 고정 소스744개, 중력보상18관절, 메모리2,791MiB와 기존Drive의
03:20 업로드·검증을 확인했다. TRAIN384·초기/학습 후DEV각128·replay25만의
작은 비교이며, 나머지 일곱 장기 실행의 TRAIN6,144·replay200만 설정은 그대로다.
원래 첫 다섯 배치와 각 배치 여섯 위치/크기 조합을 유지했다. 전체평가에는
접촉 보조를 사용하지 않는다. 실제 평가 모델 보존·초기 전체 결과·첫 TRAIN·
학습 후 전체 결과를 읽는 CPU 관찰 작업 네 개도 시작했다. 새 물리 성능,
중형/상단 개선, 독립FINAL 성공은 아직 확인하지 않았다.
[실제 시작과 백업](assets/rl_v2_URDF_precise_feedback_SAC_actual_startup_20261009.json),
[실제 CPU 관찰 작업](assets/rl_v2_URDF_precise_feedback_actual_CPU_observers_20261009.json),
[여덟 writer 실제 상태](assets/rl_v2_eight_SAC_actual_status_20261009.json).

03:52 KST에 v3의 전체 초기평가가 완료돼 성공11·안전 위반48·시간 초과56·
초기 무효13/128이었다. 실제 모델/정규화75개가 준비한 초기 모델과 같고
257개 tensor는 유한했다. Actor/Q0인 초기 기준이다. 이후 실제 TRAIN 수집과
Q 갱신을 확인했다. 선택된 경로의 접촉과 학습 후 greedy 전체평가는 대기 중이다.
[실제 초기 전체결과·같은 모델](assets/rl_v2_URDF_precise_feedback_first_full_DEV_20261009.json).

## 실제 보상의 성공·실패 순서

완료된 v2 첫 TRAIN128의 실제 보상과 종료 mask를 대조했다. 보상 shaping과
Q의 discount는 같은0.999이며 timeout도 Q에서 absorbing terminal이었다.
Held 구간 시작의 할인 return 중앙값은 성공10경로가5.95, timeout79경로가
−1.19, unsafe39경로가−5.04였다. 이 수집에서는 timeout이 성공보다 유리한
순서가 관측되지 않았다. 이는 서로 다른 실제 경로의 비교이며 같은 상태에서
동작을 바꾼 인과 비교나 다른 보상 결함이 없다는 증거는 아니다.
보상은 유지하고 안전 접근·축 정렬·실제 접촉 경험과 학습 후 평가를 계속 확인한다.
[실제 return·discount·종료 mask 대조](assets/rl_v2_completed_TRAIN_actual_returns_20261009.json).

## 장기 실행 저장 공간

종료된 우리 원본/이전 장기 실험612개 파일·31.153GiB를 디스크의
`artifacts/rl/closed_local_runs/`에 보관했다. 양쪽 모든 파일SHA256과
writer/supervisor 종료·exit0·기존 최종Drive 검증을 확인하고 원래 디렉터리
경로를 보관 위치에 연결했다. RawHDF·replay·checkpoint·영상·로그 모두 남아
있다. 데이터 총량을 보존하는 저장 위치 이동이며 활성 학습 입력은 그대로다.
00:04 KST의root 여유는약122.3GiB, `/dev/shm` 여유는약211.3GiB였다.
활성 작업으로 이후 크기는 변할 수 있다.
[원본 이동 검증](assets/rl_v2_closed_original_all6_disk_relocation_20261008.json),
[이전 장기 실행 이동 검증](assets/rl_v2_closed_tail19_disk_relocation_20261009.json).

기존Drive 연결로300초 업로드·크기/MD5 검증·오래된 검증 checkpoint 정리와
최근2개 보호를 유지한다. 현재 rawHDF/replay는 로컬 보관 범위이며 자동
삭제되지 않는다. 종료 로그는writer가 멈춘 뒤 검증한다.

## 첫 정밀 추적 수집 이후의 수정

04:17 KST의v3 첫TRAIN128은성공10·안전 위반47·시간 초과71회였다.
탐색25조건은실제파지/성공0회, greedy103조건은10회다. 닫기를시도한8경로에서
11번모두6–7틱 후12mm 거리조건으로다시열렸고 실제닫힘은35.8–52.4%였다.
축오차초과는0회이며최근접패널점도6mm 밖이었다. 관측점선택을바꾸는것만으로
해소되지않는다. 완료81,115행의명령·128종료·정수통계를대조했고, 손과flap의
움직임을랙좌표에서분리했다. 닫힘동안손이법선방향5–9mm 밀려나는사례가있어
위치추종보강을다음시험으로선택했다. finger–flap 힘0은다른부위의접촉부재를
뜻하지않으며접촉반동/drive오차의원인을확정하지않는다.

![닫기 도중 거리 증가와 미완료 그리퍼 정착](assets/rl_v2_precise_feedback_closed_motion_20261009.png)

별도v4는닫힘위치피드백2→8/s와현재flap이동량50%·틱당최대2mm 보강을쓴다.
열린접근/들기는기존속도이며모든닫기·정착·성공·안전기준을유지한다.
66검사와실제모델저장/재개/평가복원을통과했으며원래첫5배치TRAIN384·
초기/최종전체DEV128·replay25만으로비교한다. 기존장기수집은계속유지한다.
상단진입정렬과중형안전접근은아직미해결이고독립FINAL은사용하지않았다.
[실제경로와수정·선택방법](RL_V2_PERCEIVED_CONTACT_SAC_20261008.md) ·
[첫TRAIN의실제동작분석](assets/rl_v2_precise_feedback_completed_TRAIN_motion_analysis_20261009.json).

04:50 KST에새v4의실제초기전체평가·writer3,191,064·고정소스745개·
`CUDA_VISIBLE_DEVICES=3`·Kit단일GPU3·중력보상18관절·메모리2,791MiB와
첫Drive 검증04:49:56을대조했다. 기존8개실행을유지한총9개writer다.
아직actor/Q0이며TRAIN384 뒤평가효과는대기중이다. 시작기록작성오류는기존
서비스/PID를대조해기록만복구했고중복실행/프로세스신호는없었다.
[실제시작·첫검증](assets/rl_v2_URDF_motion_feedback_SAC_actual_startup_20261009.json).

04:42 KST의성공명령유지보강다섯번째전체평가는TRAIN1,920조건뒤
성공6·안전위반43·시간초과79·초기무효0/128이었다. Actor5,563/Q24,298의
실제평가전후모델/optimizer를대조했다. 자기초기14→13→8→6→5→6이며
공통유효115조건은14→5다. 중형은여전히0이므로개선으로보지않는다.
[다섯번째전체결과](assets/rl_v2_URDF_servo_guard_fifth_learned_full_DEV20_20261008.json).

04:56 KST의강한성공유지세번째전체평가는TRAIN1,152조건뒤성공8·안전
위반64·시간초과56·초기무효0/128이었다. Actor3,138/Q14,600의실제평가
전후같은모델/optimizer를대조했다. 자기초기11→10→10→8, 공통유효115조건은
11→8이며중형0이다. 초기성공수준을넘지못해장기수집과접촉추종수정을계속한다.
[강한유지세번째전체결과](assets/rl_v2_URDF_strong_gentle_full_arm_third_learned_full_DEV12_20261008.json).

05:30 KST에 v4의 전체 초기평가가 끝나 성공11·안전 위반48·시간 초과56·
초기 무효13/128이었다. 실제 초기 저장 모델/정규화와 검증한 준비 모델이
같았으며 actor/Q0인 초기 기준이다. 이후 실제 TRAIN을 시작했지만 학습 후
greedy 전체 평가의 개선은 아직 확인 전이다.
[v4 전체 초기 결과·같은 모델](assets/rl_v2_URDF_motion_feedback_first_full_DEV_20261009.json).

05:46 KST의 uniform 일곱 번째 전체평가는 TRAIN2,688 뒤 성공4·안전 위반65·
시간 초과59·초기 무효0/128이었다. 모두 상단 오른쪽small이며 중형과 다른
다섯 조합은0이다. 자기 초기14회보다 낮고 공통 유효115조건은14→3이다.
Actor7,999/Q34,044의 실제 평가 전후 같은 모델/optimizer와 고정 소스를
대조했다. 단순한 장기 학습의 개선으로 해석하지 않는다.
[일곱 번째 전체 결과·같은 모델](assets/rl_v2_URDF_regional_goal_seventh_learned_full_DEV28_20261008.json).

## 추가로 확인한 접촉 지점 후보

검증된 완료 v3 TRAIN에서 성공10경로의 첫 양손 파지, 탐색8경로의 첫 닫힘
명령, 중형 탐색6경로의 가장 가까운 상태를 비교했다. 실패한 탐색에서도
손목→TCP는 랙 안쪽을 향했으므로 반대 접근 방향이 주원인이라는 근거는
약하다. 첫 닫힘 명령에서 TCP의 flap 옆 모서리 여유 중앙값은 좌0.57mm·
우1.48mm였고, 성공 경로의 첫 실제 양손 파지는 좌18.26mm·우54.48mm였다.
왼손의 다른 접선축 여유도 실패 시1.90mm, 성공 시47.35mm였다.

이는 nominal closed TCP의 기하이며 실제 pad 접촉 위치 측정은 아니다.
닫기 시작과 실제 파지는 서로 다른 단계이므로 인과 효과를 확정하지 않는다.
기존 5mm 안쪽 목표와 6mm 닫기 허용 오차로 모서리에 가까운 닫힘이 가능해,
후속 v6는 접선 방향의 목표 여유를20mm로 늘리는 선택형 비교로 구현했다.
v4/v5의 현재 실행은 바꾸지 않으며 새 물리 성능은 아직 확인 전이다.
무작위화·안전/성공 기준은 유지한다.
[실제24경로·지점과 방향 측정·분석 범위](assets/rl_v2_actual_completed_TRAIN_panel_margins_20261009.json).

Notion에도 v5 방법 그림과 실제 CPU 대기/검증을 기록했다. 기존 영상·그림
76개가 모두 남고 새 그림을 포함한77개와 native table5개를 재확인했다.
[미디어·표 보존 검증](assets/rl_v2_upright_contact_Notion_verified_20261009.json).

## 06:01까지 완료된 추가 전체평가

관측 기반 접촉 탐색 v1은 TRAIN1,152 뒤 성공13·랙 충돌63·시간 초과52/128회였다.
자기 초기11→9→11→13, 공통 유효115조건은11→12회로 소폭 상승했으나 중형은
0이다. 5개의 기존 성공을 잃고 6개를 새로 얻었으므로 안정적인 일반화로 판단하지
않는다. 정착 v2는 TRAIN768 뒤 성공7·랙 충돌63·시간 초과58회로11→7→7이다.
각각 actor/Q3,149/14,644와1,909/9,682의 같은 실제 평가 전후 모델을 확인했다.
[v1 세 번째 전체평가](assets/rl_v2_URDF_perceived_contact_third_learned_full_DEV12_20261008.json) ·
[v2 두 번째 전체평가](assets/rl_v2_URDF_settled_contact_second_learned_full_DEV8_20261009.json).

작은 팔 탐색은 TRAIN1,536 뒤 성공6·안전 위반71·시간 초과51회로11→11→13→8→6이다.
큰 탐색은 TRAIN1,920 뒤 성공5·랙 충돌68·시간 초과55회로11→10→7→8→5→5다.
각각 actor/Q4,356/19,472와5,557/24,274의 모델·optimizer를 확인했다. 최고13회
모델은 계속 보존하지만 최신 성능은 낮아졌고 두 비교 모두 중형은0이다.
[작은 탐색 네 번째 전체평가](assets/rl_v2_URDF_gentle_full_arm_fourth_learned_full_DEV16_20261008.json) ·
[큰 탐색 다섯 번째 전체평가](assets/rl_v2_URDF_full_arm_fifth_learned_full_DEV20_20261008.json).

정밀 추적 v3는 TRAIN384 뒤 성공8·안전 위반52·시간 초과68회였다. 초기11회보다
낮고 중형은0이다. Actor702/Q4,856의 같은 모델을 확인했다. 05:53 KST에 계획을
정상 종료해 exit0·최종 checkpoint와 닫힌 로그의 Drive 검증을 마쳤다.
[v3 학습 후 전체평가](assets/rl_v2_URDF_precise_feedback_first_learned_full_DEV4_20261009.json).

## v4 완료 수집의 접촉 실패와 v5 실제 시작

v4 첫 TRAIN128은 성공10·랙 충돌40·시간 초과78회였고 탐색25조건은 성공/파지0회였다.
Actor0·Q1,590의 수집으로 학습 후 greedy 성능을 뜻하지 않는다. 검증된 완료 replay
82,554행과128종료·실행 명령을 재계산했다. 닫기를 시도한8경로는14번 모두5–7틱
뒤12mm 거리 조건으로 다시 열렸고 실제 닫힘은26.8–52.5%였다. 축 초과는0회였으며
상단 탐색11경로 모두 표면 진입 단계에 들어가지 못했다. 닫힘 추종 강화만으로
해소되지 않았다. Finger–flap 힘0을 다른 부위의 접촉 부재로 해석하지 않는다.
[첫 실제 TRAIN](assets/rl_v2_motion_feedback_first_TRAIN_actual_20261009.json) ·
[완료 replay·명령·닫힘 분석](assets/rl_v2_motion_feedback_completed_TRAIN_motion_analysis_20261009.json).

동일 완료 수집의 첫 닫힘8경로도 TCP 모서리 여유 중앙값이 좌0.57mm·우1.48mm였다.
성공10경로의 첫 양손 파지에서는 좌18.10mm·우50.53mm였다. 서로 다른 단계의
nominal TCP 기하이며 실제 pad 접촉 측정이나 원인 확정은 아니다. 20mm 접선
여유는 다음 별도 시험 후보로 유지하고 현재 v4/v5 실행에는 아직 적용하지 않는다.
[v4 실제24경로·분석 범위](assets/rl_v2_motion_feedback_completed_TRAIN_panel_margins_20261009.json).

v3의 정상 종료·최종 Drive 검증 후 v5가05:53 KST에 GPU3에서 실제 시작됐다.
06:07 KST에 writer84,979·초기 DEV0의step121·고정 소스748개·중력보상18관절을
확인했다. `CUDA_VISIBLE_DEVICES=3`, Kit단일GPU3, VRAM2,791MiB다. 06:00의
첫 백업 검증은manifest/env/agent 계약이며 초기 평가 checkpoint는 아직 저장 전이다.
기존300초 업로더가 이후 checkpoint의 크기/MD5를 확인하고 오래된 검증본만 정리한다.
대기 coordinator의exit0은 새 학습 시작 뒤 정상 종료이며 학습 중단이 아니다.
[실제 시작·설정·백업 범위](assets/rl_v2_URDF_upright_contact_SAC_actual_startup_20261009.json) ·
[실제 평가/수집 관찰 작업](assets/rl_v2_URDF_upright_contact_actual_CPU_observers_20261009.json).

기존 일곱 장기 학습의 다음 전체평가에도 GPU를 쓰지 않는 읽기 전용 관찰 작업을
실제로 등록했다. 모델·전체128결과·원래 요청·무작위화·고정 소스를 확인하며,
훈련이나 다른 사용자 프로세스에 신호를 보내지 않는다.
[다음 전체평가 관찰 PID](assets/rl_v2_next_whole_DEV_actual_CPU_observers_20261009_0611.json).

Notion 중간보고에도 실제 v5 시작과 v3 종료, 최신 v1 13회·작은 탐색6회,
v4 접촉 실패를 갱신했다. 기존 미디어77개와 native table5개가 모두 그대로
남는 것을 재확인했다.
[실제 실행 보고·미디어/표 보존 검증](assets/rl_v2_upright_contact_actual_runtime_Notion_verified_20261009.json).

## 접촉 지점을 안쪽으로 옮기는 별도 v6

06:24 KST에20mm 접선 여유를 쓰는 선택형 v6의79검사와 실제 초기 모델
저장·재개·영상용 복원을 확인했다. v5와 같은 actor/정규화·몸통 support와
frozen source를 유지했고 완료 TRAIN384상태의 초기 greedy 목표21개가 모두
정확히 같았다. 학습 Q/replay/optimizer는 새로 시작하며 이전 수집이나 분석
명령을 학습에 넣지 않는다. 원래20% TRAIN 탐색만 지점을 바꾸고 전체평가에는
보조가 없다. 완료된 v4의 실제 탐색 시작25상태에서도 새 양손 목표50개의
패널 안쪽20mm 기하 여유를 확인했으며 최대 이동은2.12cm였다.

이는 초기화·기하 검증이며 실제 성공이나 학습 개선을 뜻하지 않는다.
앞선 v4가 계획 종료와 최종Drive 검증을 마친 뒤 GPU3 메모리·디스크를 다시
확인하고 별도 비교를 이어가도록 연결한다. 기존 장기7개와 v5는 계속 유지한다.
[문제·변경·설정·검증·방법 그림](RL_V2_INTERIOR_CONTACT_SAC_20261009.md).

06:26 KST에 v6의 CPU 대기 PID589,524·빈 GPU mask·앞선 v4 writer/supervisor
생존·고정 소스749개를 실제 확인했다. 새 v6의 물리 학습은 아직 시작 전이다.
정상 종료·최종Drive 검증 뒤 GPU3에 별도 폴더로 이어지며, 초기/학습 후 전체
128결과와 같은 실제 모델을 보호한다. 현재 학습을 종료하거나 재시작하지 않았다.
[실제 대기 작업과 조건](assets/rl_v2_URDF_interior_contact_actual_GPU3_queue_20261009.json).

## 06:53 실제 학습과 추가 전체 결과

몸통 유지 v5의 초기 전체평가는06:41 KST에 성공14·안전 위반44·시간 초과57·
초기 무효13/128회로 끝났다. 실제 저장 모델/정규화77개가 검증한 초기 모델과
같고259개tensor가 유한했다. Actor/Q0이므로 학습 개선이 아니다. 중간 왼쪽
small9·상단 오른쪽small5회이고 중형과 다른 구역은0회다. 이후 첫 TRAIN을
실제로 시작했고06:53 KST에Q598회·온라인35,870행을 확인했다. Actor의 실제
갱신은 수집 warmup 이후다. 초기 checkpoint는06:45:25에Drive 검증을 마쳤다.
[전체 초기 결과·같은 모델](assets/rl_v2_URDF_upright_contact_first_full_DEV_20261009.json).

성공 명령 유지 보강의 여섯 번째 전체평가는 TRAIN2,304 뒤 성공8·안전 위반54·
시간 초과66/128회였다. 최근6→8회로 반등했지만 자기 초기14회보다 낮고 공통
유효115조건도14→7이다. 중형은0이며 랙 충돌51·박스 낙하3회였다.
Actor6,777/Q29,154의 실제 평가 전후 같은 모델·optimizer와1,084개 유한tensor를
확인했다. 장기 학습의 성공으로 표시하지 않으며 다음 TRAIN을 계속한다.
[여섯 번째 전체 결과·모델](assets/rl_v2_URDF_servo_guard_sixth_learned_full_DEV24_20261008.json).

06:53 KST에 기존 장기7개와v4/v5의9개 writer·고정 소스·GPU mask·
진행한Q 갱신을 실제 대조했다. v6는 여전히 별도CPU 대기이며 물리 학습은
시작 전이다. 현재 실행을 종료하거나 다른 사용자의 작업을 변경하지 않았다.
[현재 실행·갱신·대기 상태](assets/rl_v2_SAC_live_progress_20261009_0653.json).

Notion 중간보고에도v5 초기14회와실제TRAIN 시작, 장기 보강 비교의 최신8회를
갱신했다. 초기 결과를학습 개선으로표시하지 않으며 기존영상/그림78개와
native table5개가 모두그대로남는것을재확인했다.
[보고·미디어/표 보존 검증](assets/rl_v2_upright_initial_and_long_DEV_Notion_verified_20261009.json).

07:04 KST 추가 확인: v5가06:54에 첫 TRAIN 중 몸통 IK의 속도 제한된 복구
위치를 정책 목표 범위 위반으로 처리해 종료한 것을 확인했다. 오류 저장본과
닫힌 로그의 최종Drive 검증은 완료됐다. 그 환경의 마지막 유효 목표를 유지하고
팔 접근을 계속하는 수정과81검사·실제 관측 상태 재계산이 통과했다. 아직 실행 전
v6의 우리 CPU 대기만 중단했으며 기존 GPU 학습은 유지했다. 초기14회는 여전히
Actor/Q0 결과이고 학습 후 성공률은 없다. 수정 소스로 새로 실행한다.
[종료 원인·복구·검증](RL_V2_UPRIGHT_CONTACT_SAC_20261009.md#0704-몸통-ik-경계-오류와-복구-수정).

07:10 KST에 수정 소스3740689·749개Python 파일을 고정한 별도 장기 실행이
시작됐다. writer1,261,376·supervisor1,261,247의 소유자·실제 실행 폴더·
`CUDA_VISIBLE_DEVICES=3`를 확인했다. 기존 초기 actor/정규화와Q를 제외한
37개model tensor·몸통 source를 그대로 유지했고 Q와replay는새로 시작했다.
원래65배치·여섯 조합·TRAIN6,144·DEV각128·replay40만이며 모든17개
DEV 모델을 보존한다. 실제 초기/학습 후 전체 결과와첫 checkpoint 업로드는
아직 대기 중이다. 기존Drive about 인증은 정상이며300초 검증·최근2개 보호를 쓴다.
오류를 공유하던v6도수정 소스로 CPU 대기PID1,283,615를 새로 등록했다.
아직GPU 학습 시작 전이며v4의 정상 종료·최종Drive 검증을 기다린다.
[실제 수정 실행·관찰 작업·백업 범위](assets/rl_v2_URDF_upright_repaired_long_actual_launch_20261009.json).

강한 성공 유지의 네 번째 전체평가는 TRAIN1,536 뒤 성공6·안전 위반56·
시간 초과66/128회였다. 자기 초기11회보다 낮고 중형은0이다. Actor4,358/
Q19,478의 같은 실제 평가 모델을 확인했다. 기존 장기 학습은 계속하며
최신 감소를 개선으로 보고하지 않는다.
[강한 유지 네 번째 전체 결과](assets/rl_v2_URDF_strong_gentle_full_arm_fourth_learned_full_DEV16_20261008.json).

07:17 KST에 수정 writer의 실제 초기 DEV0 step1과중력보상18관절·초기
계약의 첫Drive 검증07:17:31을확인했다. Actor/Q0이고새전체결과는아직없다.
첫checkpoint업로드를확인한것은아니며초기평가가끝난뒤저장된다.
Notion에도옛실행의종료원인·복구수정·새장기설정·실제PID·후속대기를갱신했다.
기존미디어78개와native table5개가모두보존됐고초기14회를새학습개선으로
표시하지않는것을확인했다.
[실제 수정 보고·미디어/표 보존 검증](assets/rl_v2_upright_repaired_long_Notion_verified_20261009.json).

07:19 KST의CPU 시작검증도완료해실제writer·749개고정소스·TRAIN6,144·
17개전체DEV/65배치·replay40만·중력보상18관절·기존Drive계약검증을대조했다.
07:22 KST GPU3의실제VRAM여유는3,749MiB였다. 현재초기평가중이며새학습
후성공률을뜻하지않는다.
[실제 시작·설정·백업 검증](assets/rl_v2_URDF_upright_contact_repaired_SAC_actual_startup_20261009.json).
