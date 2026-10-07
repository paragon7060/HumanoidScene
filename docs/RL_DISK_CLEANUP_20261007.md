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
기존 Drive 원본을 읽어 검증한다.9개 관련 테스트를 통과했다.

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
