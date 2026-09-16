#!/usr/bin/env python3
"""Record 10 seconds of an unchanged native SAC checkpoint in the shared scene."""

import json
import sys
from types import ModuleType


def evaluate(env, args, directory, state):
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from cpu_scene_video import SceneVideo
    from kuavo_isaaclab_scene.rl.algorithms.sac import SAC, SACConfig

    assert env.num_envs == 1 and state['algorithm'] == 'sac'
    agent = SAC(state['obs_dim'], state['action_dim'], SACConfig(**state['config']), env.device)
    agent.restore(state, training=False)
    obs = env.reset()[0]['policy']
    renderer = SceneVideo(env, caption=f"SAC checkpoint {state['iteration']} | actual PhysX rollout | CPU mesh visualization")
    frames = 0
    measurements = []
    with imageio.get_writer(str(directory/'policy.mp4'), fps=5, codec='libx264',
                            quality=7, macro_block_size=16) as writer, torch.no_grad():
        for step in range(300):
            action = agent.act(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict['policy']
            t = env.command_manager.get_term('workcell')
            measurements.append({'step':step+1,'reward':reward.item(),
                'right_distance':t.hand_target_distance[0,1].item(),
                'right_grasp':t.hand_grasp_flags[0,1].item(),
                'unsafe':env.termination_manager.get_term('unsafe')[0].item(),
                'action':action[0].tolist(),
                'arm_tracking_error':(env.action_manager.get_term('upper_body').processed_actions
                    - t.robot.data.joint_pos[:,env.action_manager.get_term('upper_body')._joint_ids])[0].tolist()})
            if step % 6 == 5:
                frame = renderer.frame(env,step+1,measurements[-1]['right_distance'],measurements[-1]['right_grasp'])
                if frame.size == 0 or frame.max() == 0:
                    raise RuntimeError('Renderer returned an empty/black frame')
                pic = Image.fromarray(frame)
                writer.append_data(np.asarray(pic)); frames += 1
                if frames == 1 or frames == 25:
                    pic.save(directory/'preview.png')
            if (terminated | truncated).any():
                break
    report={'checkpoint':str(args.checkpoint),'iteration':state['iteration'],
            'policy':'deterministic','renderer':'CPU USD meshes at live PhysX poses',
            'frames':frames,'fps':5,'video':'policy.mp4',
            'measurements':measurements}
    (directory/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'[VIDEO] {directory / "policy.mp4"}; {frames} frames',flush=True)


if __name__ == '__main__':
    callback = ModuleType('kuavo_isaaclab_scene.rl.runners.evaluate_alternative')
    callback.evaluate = evaluate
    sys.modules[callback.__name__] = callback
    from kuavo_isaaclab_scene.rl.runners.alternatives import main
    main()
