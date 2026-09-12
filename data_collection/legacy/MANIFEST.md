# Archived on 2026-09-12

| Archived path | Reason / active replacement |
|---|---|
| `task1_cumotion_collision_plan.py` | Sequential single-arm planner; shared geometry moved to `data_collection/task1/collision.py`. |
| `task1_cumotion_replay_video.py` | Kinematic browser replay; replaced by physical `data_collection/task1/execute.py`. |
| `scripts/task1_pregrasp_smoke.py` | Pregrasp-only Isaac smoke; replaced by the canonical endpoint, approach, and physical execute pipeline. |
| `tests/task1_wrist_schedule_test.py` | Historical contract test for the archived pregrasp-only runner; retained outside active pytest discovery. |
| `configs/` | Early draft YAMLs were not the verified run contract. |
| `collection/`, `robot/`, `tasks/` | Early planning notes superseded by `data_collection/README.md` and canonical code. |
| `references/` | Original prompt/reference material retained verbatim for audit only. |

Git history before this archive is available at commit `544ed5b`.
