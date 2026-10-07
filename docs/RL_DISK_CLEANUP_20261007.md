# 종료된 우리 RL 기록의 로컬 디스크 정리

10/07 사용자의 요청으로 **26.717GiB**의 실제 디스크 블록을 정리했다.
직후 일반 사용자 디스크 여유는 **31.623GiB**였다. 서버 전체의 다른 쓰기로
현재 여유는 바뀔 수 있다.

우리 소유의 과거 종료 실험5개에서 replay5개·HDF4개, 총9개만 로컬 삭제했다.
관리자의 종료·최종 백업 확인에 더해 기존 Drive 원본의 **크기와 MD5를 직접
재검증**했다. 정리 직전 파일이 바뀌지 않았고 우리 프로세스의 열린 FD나
실행 입력에서 참조되지 않음을 확인했으며 실행별 백업 잠금도 확보했다.

| 종료 실험 폴더 | 정리한 파일 |
| --- | --- |
| `batch_sac_20261006_011850_13d57f` | replay·HDF |
| `batch_sac_20261005_230535_2b5745` | replay·HDF |
| `batch_sac_20261005_034139_08913c` | replay, HDF는 원래 없음 |
| `batch_sac_20261005_183701_e3b40a` | replay·HDF |
| `batch_sac_20261006_024149_31996b` | replay·HDF |

현재 학습·native seed2개·VR 시연·보존 정책·체크포인트·로그·평가 영상은
유지했다. 다른 사용자 파일·프로세스와 Drive 원본은 건드리지 않았다.
새 원본 데이터 업로드나 원격 삭제도 수행하지 않았다.

각 과거 실행 폴더의 `local_payload_cleanup.json`에 원래 파일명·크기·MD5와
복구 위치를 기록했다. 해당 실행을 재개하거나 원본 경로를 분석하려면
필요한 파일을 기존 인증으로 먼저 내려받는다.

```bash
export RL_DRIVE_REMOTE_ROOT='<existing-remote>:HumanoidScene-RL'
bash scripts/rl/gdrive.sh copyto \
  "$RL_DRIVE_REMOTE_ROOT/<run-directory>/<file>" \
  '/absolute/original/run-directory/<file>' --checksum --immutable
```

이는9개 파일에 한정한 정리이며 자동으로 모든 과거 기록을 삭제하는 규칙을
추가한 것이 아니다. [기존 Drive 운영](RL_GOOGLE_DRIVE.md)의 검증된 checkpoint
최신2개 보존과 활성 로그 보호는 유지한다. 같은 날 앞서 닫힌 과거 파일4개를
RAM으로 옮겨 해제한16.82GiB는 이26.717GiB에 포함하지 않는다.

사용자가 추가로 지속 정리를 요청해, 같은 검증 규칙으로 다른 종료 실험8개에서
13개 payload를 더 정리했다. **추가11.243GiB, 두 차례 로컬 삭제 합계37.960GiB·
22개 파일·13개 과거 실행**이다. 이후 서버 여유는 다시 측정하며 다른 작업의
쓰기·정리도 있으므로 여유 변화 전체를 이번 삭제량으로 주장하지 않는다.

## 진행 중인 정리 작업

[prune_verified_payloads.py](../scripts/rl/prune_verified_payloads.py)는 명시된
실험만 점검하는 선택형 CPU 작업이다. 인증을 새로 만들거나 업로드하지 않고
기존 Drive 원본을 읽어 검증한다. 현재13개 관련 테스트를 통과했다.

- writer·supervisor가 종료했고 관리자의 최종 백업 검증이 끝나야 한다.
- 현재 같은 사용자의 실행 명령·열린 FD와 명시한 보호 입력을 확인한다.
- 백업 잠금·파일 소유자·regular-file·변경 여부·원격 크기·MD5를 검사한다.
- checkpoint·로그·영상·미검증 파일은 남긴다. checkpoint·로그 전용 백업의
  replay/HDF는 원격 백업 완료로 간주하지 않아 이 작업에서 제외한다.
- 삭제한 파일마다 실행 폴더에 MD5·크기·복구 폴더를 기록한다.

실제 worker1047840을 `CUDA_VISIBLE_DEVICES=''`로 실행했고300초마다 점검한다.
지정 범위는 이번 종료 실험8개와 원래 전체 payload 백업을 수행하는 두 비교다.
현재 두 비교의 supervisor가 아직 동작 중이어서 첫 점검에서 정리를 건너뛰었다.
현재 native seed2개와 VR demo는 명시적으로 보호했다. 명단 밖 실행을 자동
발견해 삭제하지 않는다.

호스트 로컬 설정 파일에는 다음과 같이 절대 경로만 명시한다. remote 별칭과
인증은 기존 호스트 설정에서 가져오며 Git에 기록하지 않는다.

```json
{
  "experiment_dirs": ["/absolute/path/to/owned-managed-experiment"],
  "protected_paths": ["/absolute/path/to/needed-seed.hdf5"]
}
```

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/rl/prune_verified_payloads.py \
  --config /absolute/path/to/local-cleanup-config.json --watch 300
```

설정 파일 옆 `.status.json`은 마지막 점검 결과, `.lock`은 중복 worker 방지용이다.
정리는 학습 재시작·GPU 점유 변경·다른 프로세스 종료를 수행하지 않는다.

## 오래된 관리자 형식의 추가 정리

이전 reference 관리자는 `launch.json.run` 대신 `status.json.run_dir`와
동일한 원래 명령에 실행 경로를 기록했다. 이 형식도 두 명령이 정확히 같고
`--output-dir`가 기록된 실행 폴더와 일치할 때만 허용한다. 실행 경로 충돌·
명령 변경을 거부하는 검사를 추가했으며 나머지 종료·백업·FD·잠금·MD5
조건은 유지한다. 폴더 이름으로 실행 경로를 추측하지 않는다.

추가 점검은 현재 보호 입력을 보존하고, 이전 두 base/box 비교의 닫힌
reference 실행들과 명시한 종료 실험에 한정했다. 현재 사용하지 않는
replay/HDF만 지우며 체크포인트·로그·영상은 남긴다. 원격 검증에 실패하거나
프로세스가 참조하는 파일은 정리하지 않는다.

첫 추가 점검은175개 종료 실행에서274개 파일 **15.277GiB**를 삭제하고
정상 종료했다. 앞의22개 파일37.960GiB와 합하면 **296개 파일53.237GiB**다.
같은 검증으로 다른11개 과거 배치 비교의257개 종료 실행을 점검해256개에서
370개 파일 **9.801GiB**를 더 삭제하고 정상 종료했다. 따라서 이번 두 추가
점검은 **644개 파일25.077GiB**, 앞의 정리까지 오늘 로컬 디스크 삭제 합계는
**666개 파일63.037GiB**다. 검사에 실패한 과거 실행1개는 보존했다.
두 추가 점검과300초 worker의 실험 명단은 서로 겹치지 않는다.

18:00 KST에 다시 측정한 서버 일반 사용자 여유는약166.2GiB,
우리 `artifacts/rl`은약38GiB였다.
이번 요청을 다시 확인한17:18에는 우리 폴더가약63GiB였으므로 실제로 줄었다.
서버 여유는 다른 작업의 변경도 포함하므로 여유 증가 전체를 이번 삭제량으로
계산하지 않는다. 미백업 준비 입력·최신 정책·로그·영상은 계속 보존한다.

별도로 기존 성공 명령 유지 실험의 원래 전체 백업이 끝나고 writer와
supervisor가 모두 종료된 뒤 RAM replay/HDF2개도 검증하고9.960GiB를 정리했다.
이는 루트 디스크가 아니라 RAM 공간이므로 위 디스크 삭제 합계에서 제외했다.
