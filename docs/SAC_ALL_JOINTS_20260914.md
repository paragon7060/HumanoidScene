# 2026-09-14 GPU 3 전체 관절 SAC

공통 `--action-space all-joints`로 단일 medium box의 오른손 flap 파지 후
6 cm lift / 0.5초 hold를 새로 학습한다. 왼손 보조를 허용하며 현재 승인된
충돌 비용/실패 비활성, 물리 접촉 유지, curriculum 없음 설정을 상속했다.
8차원 기존 SAC 모델을 load하지 않는다.

- GPU: 물리 3, `CUDA_VISIBLE_DEVICES=3`, 학습 `cuda:0`, Kit GPU 3, multiGpu off.
- Action: base 3 + upper_body 15 + height 3 + head 2 + 양쪽 gripper 2 = 25.
- Observation: 실제 manifest 기준 269차원. base pose와 모든 integrator 포함.
- Env: 4096, spacing 8 m, dt 1/120, decimation 4.
- Replay: GPU 3000만 transition, 실제 할당 63.060 GiB. checkpoint에 replay 저장 안 함.
- Batch 1024, vector step당 update 4, rollout 16, 1500 iteration.
- Warmup: action-enabled transition 200만. settling 데이터 제외.
- 저장: 50 iteration마다 및 마지막. 300초마다 Drive 업로드/크기·MD5 검증.
- 검증된 오래된 checkpoint만 정리; 형식별 최근 2개와 검증된 최근 2개 보호.
- GPU 자체 상한 77824 MiB, 전체 free 2048 MiB 보호; 디스크 free 5 GiB 보호.
- 최대 실행 시간 86400초. 종료 후 닫힌 로그도 자동 업로드·검증.

관리 폴더:
`artifacts/rl/drive_runs/sac_gpu3_all_joints_20260914_210811_16ec03a7`

실제 실행 폴더:
`sac_20260914_210825_1f4e24`

Drive: `seonho:HumanoidScene-RL/sac_20260914_210825_1f4e24`.
기존 인증을 재사용했다. 50GB 예약 파일과 다른 사용자 프로세스를 유지했다.

실행 소스:
`artifacts/rl/source_snapshots/sac_all_joints_20260914_210811_16ec03a7`

snapshot에는 Python/config/launcher를 복사하고 SHA256 manifest를 기록했다.
대형 USD assets는 원래 asset 폴더를 symlink로 재사용한다.
관리 폴더 옆 `.launch.json`과 `.source.diff`에 Git e92b3d9 기준 실행 명령과
수정 내용을 기록했다. Drive 인증을 snapshot으로 복사하지 않았다.

시작 확인에서 실제 SAC 데이터 수집, 25 action / 269 observation,
GPU 3 단독 사용, 약 74041 MiB VRAM 및 메타데이터 백업 검증을 확인했다.
CPU 테스트 68개와 Python syntax / shell checks가 통과했다.
학습 성능 검증은 아직 수행하지 않았다. 현재 상태는 관리 폴더의
`status.json`, 진행률은 실행 폴더의 `metrics.jsonl`, 메모리는
`resources.jsonl`, 시작 근거는 `startup_audit.json`에서 확인한다.
이 문서의 실행 기록만으로 이후에도 실행 중이라고 판단하지 않는다.

공통 옵션과 파일 구조는 [RL action space 선택](RL_ACTION_SPACES.md)을 참고한다.
