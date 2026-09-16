# RL 결과를 Google Drive에 보관하기

2026-09-09 GPU 2의 새 SAC 데이터/보상 수정 실행은
[SAC 데이터·보상 변경 기록](SAC_DATA_REWARD_20260909.md)을 참고한다.
4096 env / replay 2200만 / curriculum 없음 / 50 iteration 저장 설정이며,
실행 여부는 해당 기록에 연결된 관리 폴더의 현재 상태로 판단한다.

GPU 학습과 Isaac Sim/Conda는 로컬에서 실행한다. 별도 CPU 프로세스가
완료된 체크포인트를 `gdrive:HumanoidScene-RL/<run-directory>/`에 업로드한다.
업로드 후 크기와 MD5가 일치한 파일만 로컬 정리 대상으로 삼는다.
Drive 파일을 삭제하거나 공유 권한을 변경하지 않는다.

## 다른 작업·스레드에서 기존 연결 재사용

같은 서버와 OS 사용자(`seonho`)로 작업하면 기존 인증을 재사용할 수 있다.
프로젝트 루트 [AGENTS.md](../AGENTS.md)에도 공통 안내를 기록했다.
다른 checkout/worktree에서 작업할 때는 아래 **기존 checkout의 절대 경로**를
사용하면 인증 파일을 복사할 필요가 없다.

```bash
bash /home/seonho/HumanoidScene/scripts/rl/gdrive.sh about seonho:
python3 /home/seonho/HumanoidScene/scripts/rl/drive_backup.py \
  --run-dir /absolute/path/to/unique-run-directory \
  --remote-root seonho:HumanoidScene-RL --watch 300 --keep 2
```

각 실험은 서로 다른 실행 폴더 이름을 사용한다. 원격 하위 폴더 이름은
`--run-dir`의 마지막 폴더명으로 결정되므로 서로 다른 로컬 경로라도 이름이
같으면 원격에서 충돌할 수 있다. 각 로컬 실행 폴더에 업로더 하나만 실행한다.
이미 열려 있는 스레드에는 이 문서를 읽도록 안내하면 된다. 별도 서버·다른
OS 사용자에게 인증이 자동으로 전달되지는 않는다.

## 로컬 공간은 얼마나 사용하는가

현재 방식은 마운트가 아니라 **로컬 저장 → Drive 업로드 → 검증된 오래된
체크포인트 정리**다. 업로드만으로 로컬 파일이 없어지지는 않는다.

| 항목 | 로컬 보관 방식 |
|---|---|
| 최신 체크포인트 | 각 형식별 최신 2개와 검증된 최신 2개 보호; 보통 2개, 일시적으로 더 많을 수 있음 |
| 오래된 체크포인트 | 업로더 실행 시 원격 크기·MD5 검증을 통과한 파일만 삭제 |
| TensorBoard·console 로그 | 계속 로컬에 남음; `--finished`도 로그를 업로드만 하고 삭제하지 않음 |
| Isaac Sim·conda·asset·캐시 | 그대로 로컬에 남음 |
| Drive의 과거 체크포인트 | 로컬로 자동 재다운로드하지 않음; 재개할 때 필요한 파일만 내려받음 |

2026-09-08 첫 업로드에서는 체크포인트가 `model_0.pt` 하나뿐이어서 로컬
삭제가 없었다. 따라서 그 업로드로 확보한 로컬 공간은 **0 bytes**다.
RL 결과 폴더는 확인 당시 약 296 MiB였으며, 서버 전체 공간 부족을 해결하려면
그 밖의 캐시·로그·데이터 사용량도 별도로 확인해야 한다.

## 최초 인증

이 서버에는 공식 배포 SHA256을 확인한 rclone v1.75.1을
`.external/rclone/rclone`에 설치했다. 다른 서버에서는
[공식 설치 안내](https://rclone.org/install/)에 따라 Linux 바이너리를 같은 위치에 설치한다.
이 바이너리와 인증 설정은 Git에 포함되지 않는다.

서버 터미널에서 저장소 루트를 현재 디렉터리로 두고 실행한다.

```bash
bash scripts/rl/gdrive.sh config
```

- 새 remote 이름: `gdrive`, storage 종류: `drive`.
- 내 드라이브에 새 전용 폴더를 만들 때 scope는 `drive.file`을 선택한다.
  이 권한은 해당 rclone OAuth 앱이 만든 파일과 폴더에만 접근한다.
- 기존 폴더/공유 드라이브를 사용하려면 접근 권한과 root folder 설정을 별도로 확인한다.
- 공식 문서는 공용 OAuth client ID의 2026년 지원 종료를 안내하므로,
  장기 운영에는 [직접 만든 OAuth client ID](https://rclone.org/drive/#making-your-own-client-id)를 권장한다.
- Google 로그인은 브라우저에서 직접 완료한다. 이 앱의 Drive 커넥터 인증과
  서버 rclone 인증은 별개다. 비밀번호·토큰을 채팅에 보내지 않는다.
- 서버 브라우저를 사용할 수 없다면 [원격 인증 안내](https://rclone.org/remote_setup/)에 따라
  본인 PC에서 인증하고 결과를 서버의 rclone 프롬프트에 직접 입력한다.

인증은 `.external/rclone-auth/rclone.conf`에 저장되며 디렉터리 700/파일 600
권한으로 제한된다. `gdrive.sh`는 해당 설정만 사용한다.

```bash
bash scripts/rl/gdrive.sh about gdrive:
bash scripts/rl/gdrive.sh mkdir gdrive:HumanoidScene-RL
bash scripts/rl/gdrive.sh lsd gdrive:
```

사용자가 약 4TB 사용 가능하다고 알려 주었으며, 실제 계정 용량은 인증 후
`about`으로 확인한다. Shared Drive에서는 계정 용량이 보고되지 않을 수 있다.

### 이 서버의 실제 연결 (2026-09-08)

인증한 remote 이름은 `seonho`다. 이 서버에서는 위 명령의 `gdrive:`를
`seonho:`로 바꾸고, 업로더에 `--remote-root seonho:HumanoidScene-RL`을
명시한다. 기본값 `gdrive:`는 자동으로 계정을 선택하지 않는다.
`about seonho:`로 인증을 확인했으며 총 5 TiB, 사용 가능한 공간은
5,489,515,262,953 bytes (약 4.99 TiB)였다. 용량은 확인 시점의 값이다.
`HumanoidScene-RL/train_20260907_223932_c32c48`에 기존 실행의 메타데이터
4개, `model_0.pt`, TensorBoard event 파일을 업로드하고 각각 크기/MD5를
검증했다. 로컬 파일은 모두 유지했다. 상시 업로더와 학습 재시작은 아직 하지 않았다.

## 종료된 학습의 결과 업로드

`--run-dir`는 `manifest.json`이 들어 있는 **개별 실행 폴더**다.

```bash
python3 scripts/rl/drive_backup.py \
  --run-dir artifacts/rl/stable_grasp/train_20260907_223932_c32c48 \
  --remote-root seonho:HumanoidScene-RL \
  --finished
```

PPO의 `model_<iteration>.pt`, SAC/DPPO의 `checkpoint_<iteration>.pt`와
manifest/env/agent/verification 메타데이터를 업로드한다. `--finished`는 종료된
실행 폴더의 TensorBoard event, metrics.jsonl, `.log` 파일도 포함한다.
실행 폴더 밖의 console 로그, 데이터셋, 소스, 인증 파일은 포함하지 않는다.
이미 다른 내용의 파일이 같은 원격 경로에 있으면 덮어쓰지 않고 실패한다.

## 학습 중 별도 업로더 실행

```bash
python3 scripts/rl/drive_backup.py \
  --run-dir /absolute/path/to/current-run \
  --remote-root seonho:HumanoidScene-RL \
  --watch 300 --keep 2
```

5분마다 확인하며 수정 후 최소 120초가 지난 체크포인트를 업로드한다.
체크섬 검사와 업로드 전후 파일 상태 검사를 통과해야 정리한다.
각 체크포인트 형식별로 최신 2개 및 검증된 최신 2개를 보호하므로,
아직 저장 중인 파일이 있으면 일시적으로 2개보다 많이 남을 수 있다.
원격에는 업로드한 과거 체크포인트를 유지한다.
같은 실행 폴더에 업로더 두 개를 실행하면 파일 잠금으로 두 번째를 거부한다.

한 번 실행할 때는 실패 시 nonzero로 종료한다. `--watch`에서는 실패한 회차의
체크포인트 정리를 건너뛰고 다음 회차에 재시도한다. 네트워크 장애로 학습
프로세스가 직접 중단되지는 않지만, 업로드가 실패하면 로컬 공간은 계속 필요하다.

## 현재 범위와 남은 작업

- 인증·실제 Drive 업로드 검증 전에는 연결 완료로 간주하지 않는다.
- 실행 중인 로그는 업로드하지 않는다. 단독 업로더는 종료 후 `--finished`를 사용한다.
  아래 PPO 관리자는 종료 후 업로드와 디스크 부족 시 종료를 연결한다.
  로그 회전, OS 서비스 등록, 중단 순간의 추가 체크포인트 저장은 제공하지 않는다.
- `drive_backup.py` 자체는 저장 간격을 바꾸지 않는다. PPO의 현재 기본 1000 iteration은
  초기에 중단되면 복구 손실이 크므로, 업로드 확인 후 간격을 줄여 운영할 수 있다.
- 일반 SAC/DPPO 실행의 기본 보존은 로컬 최근 2개다. Drive와 함께 실행할 때는
  `--external-checkpoint-retention`으로 학습기의 삭제를 끈다. 아래 SAC/DPPO
  관리자는 이 옵션을 자동 지정하므로 검증된 업로더만 오래된 체크포인트를 삭제한다.
- rclone 마운트의 쓰기 캐시는 로컬 공간을 사용하고, 열린 파일은 캐시에서
  제거할 수 없다. 지속 기록하는 TensorBoard 경로를 마운트로 바꾸지 않는다.
  [rclone 캐시 동작](https://rclone.org/commands/rclone_mount/#vfs-file-caching)
- 서버 전체 디스크가 가득 차는 문제는 별도로 해결해야 한다. 학습 결과의
  클라우드 보관만으로 Isaac 캐시나 다른 사용자의 디스크 사용량까지 줄어들지는 않는다.

## PPO 학습과 자동 업로드를 함께 관리하기

`scripts/rl/train_with_drive.py`는 새 실험 전용 폴더에서 PPO와 CPU 업로더를
함께 관리한다. 기존 `seonho:` 인증과 `drive_backup.py`의 검증·정리 함수를
재사용한다. OS 서비스가 아니므로 서버 재부팅 이후 자동 재시작되지는 않는다.

```bash
# 반드시 이전에 사용하지 않은 고유한 경로를 지정한다.
python3 scripts/rl/train_with_drive.py \
  --experiment-dir /absolute/path/to/unique-ppo-experiment \
  --remote-root seonho:HumanoidScene-RL \
  --num-envs 16384 --max-iterations 2000 --save-interval 4 \
  --checkpoint /absolute/path/to/previous-run/model_0.pt
```

- `CUDA_VISIBLE_DEVICES=1`, 학습 장치 `cuda:0`, Kit 렌더러의 물리 GPU 1을
  지정하고 다중 GPU 렌더링을 끈다. 관리자 자체는 GPU를 사용하지 않는다.
- 실험 폴더 아래 `train_<timestamp>_<uuid>`가 실제 실행 폴더이며 동일한
  이름을 Drive 하위 폴더로 사용한다. 학습 console도 이 폴더 안에 저장한다.
- 기본 저장 간격은 이 관리자 실행에 한해 10 iteration이다(2026-09-11 변경). 업로드는 300초마다
  검사하며, 최소 120초 전에 저장을 마친 체크포인트를 크기·MD5로 검증한다.
  최신 2개와 검증된 최신 2개를 보호하므로 업로드 대기 중에는 더 남을 수 있다.
- PPO는 임시 `.pending` 파일에 저장한 뒤 완료 시 `model_<iteration>.pt`로
  원자적으로 이름을 바꾼다. 중단된 저장 파일을 업로더가 완료 모델로 취급하지 않는다.
- 로컬 여유 공간이 5 GiB 미만이면 이 관리자가 생성한 학습 프로세스 그룹에만
  종료 신호를 보낸다. 120초 동안 끝나지 않으면 해당 그룹만 강제 종료한다.
  다른 사용자 파일이나 프로세스는 정리하지 않는다.
- 학습 프로세스가 끝나고 console이 닫힌 뒤, 진행 중이던 업로드가 끝날 때까지
  기다리고 최종 체크포인트·TensorBoard·console 로그를 업로드하고 검증한다.
  최종 업로드 실패 시 로컬 파일을 보존하고 300초마다 재시도한다.
- 실험 부모 폴더의 `status.json`에 실제 PID, 실행 폴더, 마지막 검증 시간,
  업로드 오류 및 종료 코드를 기록한다. `final_upload_verified: true`는 최종
  업로드 검증 완료를 뜻하며, 학습 성공 여부는 `training_exit_code`로 구분한다.
  부모 폴더의 관리 로그와 상태 파일은 실행 중인 관리용 파일이므로 업로드 대상에서 제외한다.

2026-09-08 시작한 PPO 재개 실험의 관리 폴더는
`artifacts/rl/drive_runs/ppo_gpu1_drive_20260908_011530_77a6e7ff`다.
실제 실행 폴더는 `train_20260908_011537_9c2516`이며 Drive 대상은
`seonho:HumanoidScene-RL/train_20260908_011537_9c2516`이다.
이 기록 자체는 실행 상태의 증거가 아니므로 현재 PID와 `status.json`을 확인한다.

2026-09-08 17:31에는 충돌면 중점 기준점과 비양수 접근 비용을 적용한 새 PPO를
시작했다. GPU 1 / 16384 env / 227-D 관측 / 2000 iteration / 저장 간격 4다.
기존 223-D 모델을 load하지 않는 새 학습이며 관리 폴더는
`artifacts/rl/drive_runs/ppo_grasp_v2_gpu1_20260908_173131_887e0af4`, 실행 폴더는
`train_20260908_173139_7e0b1e`다. Drive 대상은
`seonho:HumanoidScene-RL/train_20260908_173139_7e0b1e`다. 자동 업로드는 300초 간격이며,
종료 뒤 로그를 검증하고 로컬 디스크 5 GiB 보호 정책을 유지한다.
`.launch.json`에 실제 명령, git revision 및 핵심 소스의 SHA256을 기록했다.
현재 실행 여부와 업로드 완료 여부는 새 관리 폴더의 `status.json` 및 실제 PID로 확인한다.

## SAC / DPPO와 GPU 메모리 제한

`sac_with_drive.py`와 `dppo_with_drive.py`는 위 `train_with_drive.supervise` 및
기존 `drive_backup.py`를 재사용한다. 새로운 인증·원격·별도 백업 프로토콜을 만들지 않는다.
고유한 부모 폴더를 자동 생성하며, 그 아래 `sac_*` / `dppo_*`가 실제 실행 폴더다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/rl/sac_with_drive.py \
  --gpu 0 --num-envs 17408 --max-iterations 6 --rollout-steps 16 \
  --replay-device cuda:0 --replay-capacity 1000000 \
  --batch-size 4096 --updates-per-step 4 --learning-starts 100000 \
  --save-interval 2 --remote-root seonho:HumanoidScene-RL
```

- `--gpu`가 물리 GPU 번호이며 자식은 해당 `CUDA_VISIBLE_DEVICES`와 `cuda:0`을 사용한다.
  기존 alternatives runner가 Kit renderer도 같은 물리 GPU로 지정한다.
- SAC 기본 replay 100만 transition은 현재 223-D/16-D 계약에서 약 1.73 GiB다.
  체크포인트에는 replay가 들어가지 않는다. `--replay-device cpu`로 RAM에 둘 수도 있다.
- 기본은 6 iteration의 짧은 테스트이며 2 iteration마다, 그리고 종료 시 저장한다.
  장기 실험의 iteration 수와 저장 간격은 별도로 지정한다. 업로드 검사 주기는 300초다.
- `--external-checkpoint-retention`을 자동 지정한다. 네트워크 오류가 나면 미검증
  파일을 삭제하지 않으므로 일시적으로 2개를 초과할 수 있다. 모든 파일을 검증한
  회차 뒤에는 최신 2개가 남고 이전 파일은 Drive에 보존된다.
- GPU 메모리는 5초마다 해당 자식 PID만 측정해 `resources.jsonl`에 기록한다.
  기본 자체 GPU 상한 34816 MiB, 전체 GPU 여유 8192 MiB, 로컬 디스크 여유 5 GiB를
  보호한다. 제한을 넘으면 이 관리자의 자식 프로세스 그룹만 종료한다.
  `--max-seconds` 기본 3600초는 초기화 시간을 포함한 테스트 상한이다.
- 학습 종료 후 `metrics.jsonl`, `resources.jsonl`, `status.json`, console 로그와
  체크포인트를 크기·MD5로 검증한다. 부모 `status.json`의
  `final_upload_verified: true` 및 `training_exit_code: 0`을 함께 확인한다.
  Isaac Kit가 예외를 exit code 0으로 덮어도 실행 폴더의 실패 상태를 확인해 실패로 기록한다.

DPPO는 동일 관리자 구조에서 `--checkpoint`가 필요하다. checkpoint 없이 배선·메모리만
시험할 때는 `--smoke-test`를 명시하며 최대 10 iteration으로 제한한다. 결과와
체크포인트에 `random_smoke_test_not_pretrained`를 기록해 사전학습 결과와 구분한다.
이 랜덤 초기화 테스트의 결과는 파지 성능 검증이 아니다.

### 2026-09-08 SAC GPU 0 측정

- 8192 env / 1 iteration: 최대 16868 MiB (16.47 GiB), 1464.5 transition/s.
- 17408 env / 6 iteration: 최대 **30274 MiB (29.56 GiB)**,
  평균 **2524.4 transition/s**, 총 1,671,168 transition.
- 큰 실행 설정: rollout 16, minibatch 4096, vector step당 4 updates,
  learning-starts 100000, GPU replay 1000000, save-interval 2.
  Replay가 가득 찬 뒤에도 메모리는 위 범위였고 GPU 최소 여유는 14117 MiB였다.
- 마지막 체크포인트의 모델·정규화·optimizer tensor 91개가 모두 유한했다.
  체크포인트 하나는 5,687,582 bytes이며 replay는 저장하지 않는다.
- 6회는 구동·메모리·백업 검증이다. 이 동안 완료된 episode 7개는 모두 충돌 실패였으며
  파지 성공 정책이 확보됐다는 결과가 아니다.
- 실제 실행 폴더:
  `artifacts/rl/drive_runs/sac_gpu0_20260908_014124_f5f29f72/sac_20260908_014132_099fab`.
  Drive 대상은 `seonho:HumanoidScene-RL/sac_20260908_014132_099fab`이다.
  부모의 `status.json`, `training_audit.json`으로 실행과 검증 상태를 확인한다.
- 종료 후 원격 checkpoint 2/4/6과 로그·메타데이터 총 11개 파일의 크기·MD5를
  대조했다. 로컬에는 checkpoint 4/6만 남았으며 학습·관리 PID 모두 종료됐다.
  대조 기록 `drive_audit.json`도 같은 Drive 폴더에 업로드하고 검증했다.

### GPU 2에서 SAC 학습 재개

2026-09-08 사용자 요청에 따라 GPU 0 테스트의 checkpoint 6에서 **300 iteration을
추가**하는 실행을 시작했다. 이전 모델·정규화·optimizer를 복원하며 replay는 새로 채운다.
종료 iteration은 306이다. 저장은 10 iteration 간격과 마지막 iteration이며,
Drive 검사는 300초 간격이다. `--max-seconds 86400`으로 테스트용 1시간 제한을 늘렸다.

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/rl/sac_with_drive.py \
  --gpu 2 --num-envs 17408 --max-iterations 300 --rollout-steps 16 \
  --batch-size 4096 --updates-per-step 4 --learning-starts 100000 \
  --replay-capacity 1000000 --replay-device cuda:0 --save-interval 10 \
  --max-seconds 86400 --remote-root seonho:HumanoidScene-RL \
  --checkpoint artifacts/rl/drive_runs/sac_gpu0_20260908_014124_f5f29f72/sac_20260908_014132_099fab/checkpoint_00000006.pt
```

관리 폴더는 `artifacts/rl/drive_runs/sac_gpu2_resume_20260908_032357_b27156fd`,
실제 실행 폴더는 `sac_20260908_032408_3deaf0`다. Drive 대상은
`seonho:HumanoidScene-RL/sac_20260908_032408_3deaf0`다.
앞선 `sac_gpu2_resume_20260908_031046_cd1bdbfc` 시도는 초기화 중 종료했으며
학습 업데이트 없이 console 로그 업로드·검증을 마쳤다. 종료 시 장면 생성 시간이
490초로 기록되어 교착 여부는 확인되지 않았다. 재시도에서는 체크포인트 역직렬화를
장면 생성 뒤로 옮기고 초기화 중 주기적인 Python stack 진단을 추가했다.
재시도에서 USD 환경 복제는 344초 뒤 완료됐고 이후 PhysX·접촉 센서 초기화로
진행했다. 당시 서버 CPU pressure가 높았다. 체크포인트 로딩 때문에 교착됐다고
판정할 근거는 없으며, 대규모 실행의 초기화에는 여러 분이 필요하다.
03:38 이후 checkpoint 6 복원을 마치고 iteration 7/8의 실제 업데이트를 확인했다.
손실은 유한했고, iteration 8에서 보상이 기록됐다. 프로세스의 GPU 2 단독 사용과
기존 checkpoint의 환경 계약 일치를 확인했으며 당시 GPU 사용량은 30215 MiB
(29.51 GiB)였다. 관리자가 manifest/env/agent 파일의 Drive 업로드·MD5 검증을
마쳤다. 시작 검증은 관리 폴더의 `startup_audit.json`에 기록했다.
실행 상태는 관리 폴더의 `status.json`, 학습 추세는 실행 폴더의 `metrics.jsonl`에 있다.
기존 성공·충돌률에 더해 `workcell/right_target_distance`, `workcell/grasp_right`,
`workcell/lift_height`, `workcell/hold_fraction` 등 매 iteration 마지막 step의
전체 env 평균을 기록한다. 기록의 존재만으로 현재 실행 중이라고 판단하지 않는다.

### 2026-09-08 observation / EEF 버전 2 SAC 새 학습

관측 227차원, 실제 손가락 충돌면 중점 EEF, 접근·정렬·닫기·접촉 오차 비용,
공통 그리퍼 마찰 20/16을 사용한다. 기존 223차원 checkpoint를 재개하지 않는다.
CPU 테스트 26개와 실제 Isaac 8 env / SAC 3 iteration 검증이 통과했다.
소규모 검증 관리 폴더는 `sac_gpu2_eefv2_probe_20260908_173218_096a423d`이며
실제 실행 폴더 `sac_20260908_173230_601566`에 관측 계약과 지표가 있다.

본 학습 관리 폴더:
`artifacts/rl/drive_runs/sac_gpu2_eefv2_20260908_173528_e8718eeb`.
GPU 2를 `CUDA_VISIBLE_DEVICES=2`로 지정하고 17408 env, 300 iteration,
rollout 16, batch 4096, updates-per-step 4, learning-starts 100000,
GPU replay 1000000, save-interval 10, max-seconds 86400으로 실행한다.
기존 `seonho:HumanoidScene-RL`로 300초마다 검증·업로드하며 최신 두 checkpoint를
유지한다. 종료 후 닫힌 로그까지 검증하는 기존 관리자를 재사용한다.
실제 실행 폴더와 현재 상태는 부모 `status.json`에서 확인한다.
실행 당시 소스 SHA256과 Git 차이는 부모 옆 `.launch.json`, `.source.diff`에 기록했다.
실제 실행 폴더는 `sac_20260908_173540_853152`이며 Drive에도 같은 이름을 쓴다.
17:48에 본 학습의 첫 SAC 업데이트와 유한한 손실, 30243 MiB (29.53 GiB)
GPU 사용량, 새 관측·EEF 계약 및 Drive 메타데이터 검증을 확인했다.
`startup_audit.json`은 시작 검증 기록이며 현재 진행률은 `metrics.jsonl`로 확인한다.

50GB 예약 파일 `.external/rl-storage/reserve_50GB.img`는 그대로 보존했다.
이번 실행은 일반 `artifacts` 경로를 쓰며, 예약 파일의 공간을 직접 쓰는 마운트는
아니다. 시작 전 별도 디스크 여유 약 130 GiB와 Drive 연결을 확인했다.

## 2026-09-11 PPO 정책 수정 실험

`train_with_drive.py`는 PPO 완료/실패 status.json을 확인하고, GPU 1 사용량 및
여유 메모리를 5초마다 기록한다. 기본 자체 한도 78000 MiB, 전체 GPU free 2048 MiB,
최대 실행 시간 7일이다. `--gpu-limit-mib`, `--gpu-reserve-mib`, `--max-seconds`로 지정한다.
기존 Drive 연결과 업로드 300초/최근 checkpoint 2개 보호/종료 로그 검증을 재사용한다.

현재 실험 설정과 검증 결과는 [PPO 정책 수정 및 실행](PPO_BOUNDED_RUN_20260911.md)을 참고한다.
관리 폴더는 `artifacts/rl/drive_runs/ppo_bounded_gpu1_20260911_182052_764df1`,
실제 실행 폴더는 `train_20260911_182101_98e97e`다. 16384 env / PPO 2000 iteration /
64-step rollout / 10 iteration 저장이며, 실제 진행 여부는 status.json 및 PID로 확인한다.
