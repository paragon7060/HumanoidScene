# RL 프로세스의 CUDA·그래픽 GPU 격리

`CUDA_VISIBLE_DEVICES`는 CUDA 장치를 제한하지만 Vulkan/GL 장치 선택까지
제한하지 않는다. 기존 Isaac 실행은 GPU 3에서 학습하면서 GPU 0에도 6MiB의
`G` 컨텍스트를 생성했다. `activeGpu=3`과 multi-GPU 비활성화만으로는 이
컨텍스트가 없어지지 않았다.

`scripts/rl/batched_staged_goal_with_drive.py`는 이제 기본으로
`single_gpu_runtime.py`를 사용한다. Linux의 기존 `bwrap`으로 **우리 실행에만**
선택한 GPU의 장치 파일을 노출한다. 호스트 장치, 권한, 드라이버와 다른
사용자의 프로세스는 바꾸지 않는다. 선택한 GPU UUID를 자동으로 조회해
`CUDA_VISIBLE_DEVICES`에 넣으며, namespace 안에서는 CUDA와 렌더러가
각각 `cuda:0`, `activeGpu=0`을 사용한다. 이 0은 호스트 GPU 0이 아니라
그 실행에서 유일하게 보이는 GPU다. `--gpu 3`은 계속 호스트 GPU 3을 뜻한다.

```bash
CUDA_VISIBLE_DEVICES=3 python3 scripts/rl/batched_staged_goal_with_drive.py \
  --gpu 3 --experiment-dir /absolute/path/to/unique-run \
  --checkpoint-log-backup-only \
  ...
```

실제 작업별 필수 학습 인수는 기존 실험 문서를 따른다. Drive 인증·업로더는
원래 checkout을 재사용하며 namespace에 인증을 복사하지 않는다.
Linux와 `bwrap`, NVIDIA 장치가 없으면 실행을 거부한다. 명시적인
`--no-isolate-graphics-devices`는 이전 실행 경로를 선택하지만 다른 GPU의
작은 그래픽 컨텍스트가 생기지 않는다는 보장은 없다.

2026-10-09의 짧은 실제 Isaac 테스트에서 GPU 3만 활성 상태이고 Torch의
장치 수가 1이며 CUDA tensor 연산이 정상임을 확인했다. 호스트 `nvidia-smi`에서
그 PID는 GPU 3에만 `C+G`로 표시됐고 GPU 0·1·2에는 없었다. 단순히 프로세스
표를 숨긴 것이 아니다. 기존 실행의 컨텍스트는 실행 중에 이 설정으로
바뀌지 않으므로 정상 종료 후 새 실행에 적용한다.

확인은 compute 전용 조회 대신 **전체 `nvidia-smi`**로 한다. compute 조회에는
다른 GPU의 6MiB `G` 항목이 표시되지 않는다. 실행의 `launch.json`에는 호스트
GPU와 `single_GPU_graphics_isolation`, `status.json`에는 실제 writer PID와
CUDA UUID 마스크를 기록한다. 감독 프로세스는 private namespace 안에서
writer를 생성하므로 writer PID는 호스트에서 보이는 실제 PID다.

2026-10-10 실행 조회 수정: 장치의 minor number는 이 서버의 NVIDIA 드라이버에서
`--query-gpu` 필드로 지원되지 않아 새 관리자가 시작 전에 실패했다. 이제
`nvidia-smi --id=<gpu> -q -x`의 XML inventory에서 UUID·minor number·PCI 주소를
읽는다. 호스트 GPU index와 장치 minor number가 다를 수도 있으므로 별도로
보존한다. 이 수정은 학습·보상·초기 model을 변경하지 않는다.

관련 근거: [NVIDIA의 Vulkan/CUDA 장치 선택 설명](https://docs-prod.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q9-how-to-specify-what-gpus-to-run-omniverse-apps-on).
