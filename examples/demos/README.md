# V2 staged-grasp 성공 데모 예시

Quest 3 VR 원격조작으로 수집한 **성공 에피소드 2개**(910 transitions, 30 Hz 기준 약 30초)다.
파일 크기를 위해 gzip으로 압축했을 뿐, 저장된 값은 원본과 bit 단위로 동일하다.

| 항목 | 값 |
|---|---|
| 파일 | `v2_grasp_quest_success.hdf5` (1.7 MB) |
| 원본 | `v2_grasp_quest_005.hdf5`의 `episode_000001`(389), `episode_000003`(521) |
| 추출 | `python3 scripts/rl/export_demo_subset.py <입력> --output <출력> --compression-level 9` |
| 수집 | `./quest_collector.sh collect --rl-reward-debug 2 --rl-demo-dataset ...` ([문서](../../docs/RL_QUEST_REWARD_DEBUG.md)) |

## 환경 계약

`manifest_json` 루트 attribute에 전체 설정이 들어 있다. 학습에 사용하기 전 아래가 맞는지 확인한다.

| 키 | 값 |
|---|---|
| `action_dim` | 25 (정규화 관절 증분 + binary gripper) |
| `actor_obs_dim` / `critic_obs_dim` | 403 / 469 (critic = policy + privileged) |
| `control_dt` | 1/30 s |
| `gripper_close_force_n` | 50 |
| `rack_rollers` | true |
| `multi_box.self_collision_enabled` | **false** |

self-collision을 켠 학습에 그대로 섞으면 안 된다. 보상·종료·관측은 `MultiBoxGraspAssemblyEnvCfg`에서 나온 값이다.

## 구조

```
/                      attrs: format, format_version, manifest_json, exported_from_json
└── episodes/
    └── episode_000000  attrs: success, end_reason, num_transitions, source_file, source_episode
        └── transitions/
            actor_obs, critic_obs, action, reward,
            next_actor_obs, next_critic_obs,
            terminated, truncated, success, unsafe, sim_time_s
```

`next_*`는 자동 reset 이전의 terminal 관측이므로 SAC replay에 그대로 넣을 수 있다.

```python
import h5py, json

with h5py.File("examples/demos/v2_grasp_quest_success.hdf5") as file:
    manifest = json.loads(file.attrs["manifest_json"])
    for name, episode in file["episodes"].items():
        transitions = episode["transitions"]
        print(name, episode.attrs["end_reason"], transitions["reward"].shape[0])
```

## 한계

- 사람이 조작한 시연이라 **최적 궤적이 아니다.** 접근 중 멈칫거림, 접촉 손실 후 재파지, 불필요한 몸통 이동이 들어 있다.
- 2개 에피소드는 정책을 지도학습하기에 턱없이 적다. off-policy replay의 초기 seed 용도로만 쓴다.
- 현재 V2 SAC는 `--demo-dataset examples/demos/v2_grasp_quest_success.hdf5
  --no-self-collision`으로 두 에피소드를 별도 replay에 넣는다. 기록된
  base-relative TCP/박스 자세로 403→441 actor 관측과 469→507 critic 관측을
  다시 계산하므로 Isaac 물리 재생은 필요 없다. 원본 `reward`와 terminal 전후
  관측은 유지하며, 파일 SHA256과 변환 계약을 run manifest에 남긴다.
- 기본 `--demo-batch-fraction 0.2`는 각 SAC minibatch의 20%를 이 별도
  replay에서 표본 추출한다. 시연이 2개뿐이므로 일반화나 성공을 보장하지 않는다.
  원본에는 현재 보상의 버전 해시가 없어 과거 dense reward를 정확히 재계산·
  동등성 검증할 수 없다.
