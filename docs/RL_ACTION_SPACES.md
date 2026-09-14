# RL action space 선택

PPO, SAC, DPPO의 공통 runner와 4박스 staged/end-to-end PPO에서
`--action-space right-arm|all-joints`를 사용한다. 생략하면 기존 실험 config의
action 구성을 유지한다. CLI 선택은 `configure_task()` 이후에 적용한다.

| 선택 | Policy action | 고정되는 부분 |
| --- | --- | --- |
| `right-arm` | 오른팔 7관절 + 오른쪽 그리퍼 1 = 8차원 | base, torso, head, 왼팔, 왼쪽 그리퍼 |
| `all-joints` | planar base 3 + torso 4 + 양팔 14 + head 2 + 그리퍼 2 = 25차원 (S200062) | passive linkage joint |

base는 기존 PlanarDrive의 평면 속도/회전 명령으로 root pose를 적분한다.
휠 토크로 주행을 학습하는 방식은 아니다. 그리퍼의 여러 linkage 관절은
손당 하나의 열기/닫기 명령으로 제어한다. 오른팔 모드의 비활성 관절은
reset pose를 latch하고 물리 관절 제한과 target으로 고정한다.

```bash
# PPO: 기존 flap pick config에 공통 옵션 적용
CUDA_VISIBLE_DEVICES=1 bash train_flap_pick.sh --action-space right-arm
CUDA_VISIBLE_DEVICES=1 bash train_flap_pick.sh --action-space all-joints

# SAC / DPPO: base·torso까지 허용한 단일 박스 lift config
CUDA_VISIBLE_DEVICES=3 bash scripts/rl/mobile_flap_pick.sh sac --action-space all-joints
CUDA_VISIBLE_DEVICES=3 bash scripts/rl/mobile_flap_pick.sh sac --action-space right-arm
# DPPO에는 호환되는 사전학습 checkpoint가 필요하다.
CUDA_VISIBLE_DEVICES=3 bash scripts/rl/mobile_flap_pick.sh dppo \
  --action-space all-joints --checkpoint /absolute/path/to/checkpoint.pt
```

이 서버의 기존 로컬 Drive 관리자 `train_with_drive.py`, `sac_with_drive.py`,
`dppo_with_drive.py`도 같은 옵션을 전달한다. 이 관리자들은 별도 미커밋 변경이며
action-space 브랜치에는 포함하지 않았다. SAC 관리자의 `--experiment mobile-flap-pick`은
`configs/rl_pick_whole_body.py`를 선택한다. 일반 flap config에서도
`--action-space all-joints`를 선택할 수 있다.

이동이 필요한 approach/carry/full은 오른팔 모드에서 설정 오류로 거부한다.
4박스 PPO는 staged pick/extract/place에서 오른팔 옵션을 사용할 수 있지만,
carry/full에서는 base가 필요하다. reset bank와 checkpoint는 action/관측
계약이 같아야 한다. 8차원 또는 기존 23차원 모델을 25차원 모델로 재개할 수 없다.

공통 선택/CLI/joint 이름은 `src/kuavo_isaaclab_scene/rl/action_spaces.py`,
action term 구성은 `rl/managers/actions.py`, 환경 선택은 `rl/envs/env_cfg.py`에서
관리한다. 4박스 task는 독립 manager를 유지하면서 공통 선택 이름을 재사용한다.
전체 flap pick 관측은 base pose와 모든 action integrator 상태를 포함한다.
보상과 성공 조건은 action 선택과 별개다. 새 whole-body config는 기존
오른손 flap 파지 후 6 cm lift 및 0.5초 hold, curriculum 없음, 현재 승인된
충돌 비용/실패 비활성 설정을 유지하며 왼손 보조를 허용한다.
