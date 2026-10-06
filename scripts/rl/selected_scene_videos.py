"""Record measured parallel DEV bodies before reset, without replaying physics."""
from pathlib import Path
from types import SimpleNamespace
import json


def measured_asset_view(asset, index, origin):
    """A read-only env-local pose view for the existing CPU mesh renderer."""
    data=asset.data
    fields=dict(root_pos_w=data.root_pos_w[index:index+1]-origin,
        root_quat_w=data.root_quat_w[index:index+1])
    if hasattr(data,'body_link_pos_w'):
        fields.update(body_link_pos_w=data.body_link_pos_w[index:index+1]-origin,
            body_link_quat_w=data.body_link_quat_w[index:index+1])
    return SimpleNamespace(data=SimpleNamespace(**fields))


def validate_video_selection(indices, num_envs, steps):
    if not indices:return
    if len(indices)>6 or len(set(indices))!=len(indices) or any(i<0 or i>=num_envs for i in indices) or steps!=900:
        raise ValueError('DEV video requires1..6 distinct valid environment indices and900 steps')


class SelectedSceneVideos:
    def __init__(self,env,output,wave_index,layouts,indices,*,actor_updates,critic_updates,capture_every=10):
        import cv2
        from cpu_scene_video import SceneVideo
        validate_video_selection(indices,env.num_envs,900)
        self.env,self.output,self.wave,self.indices=env,Path(output),wave_index,indices
        self.renderer=SceneVideo(env);self.meshes=self.renderer.meshes
        self.capture_every=capture_every;self.cv2=cv2;self.writers={};self.records=[];self.last_steps={}
        self.actor_updates,self.critic_updates=actor_updates,critic_updates
        for i in indices:
            region=layouts[i]['layout']['target_region']
            raw=self.output/f'eval_wave_{wave_index:04d}_env_{i:03d}_raw.mp4'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30/capture_every,(960,720))
            if not writer.isOpened():raise RuntimeError('DEV video writer could not open')
            self.writers[i]=writer
            self.records.append(dict(environment=i,wave=wave_index,region=region,raw_file=raw.name,
                layout=layouts[i]['layout'],frames=0,actual_outcome=None,
                actor_updates_at_start=actor_updates,critic_updates_at_start=critic_updates,
                role='frozen_DEV_policy_measured_body_poses_before_automatic_reset',
                simulation_device=str(env.device),capture_every_control_steps=capture_every,
                Q_or_replay_rows_imported=False,not_inferred_robot_or_flap_animation=True))

    def capture(self,step,active,success,unsafe,time_out,distances):
        """Called from termination compute while the true terminal bodies exist."""
        for record in self.records:
            i=record['environment']
            if i not in self.writers:continue
            if not bool(active[i]) or self.last_steps.get(i)==step:continue
            final=bool(success[i] or unsafe[i] or time_out[i])
            if step%self.capture_every and not final:continue
            origin=self.env.scene.env_origins[i:i+1];views={}
            meshes=[]
            for points,faces,owner,color in self.meshes:
                if owner:
                    asset,body=owner;key=id(asset)
                    if key not in views:views[key]=measured_asset_view(asset,i,origin)
                    owner=(views[key],body)
                meshes.append((points,faces,owner,color))
            self.renderer.meshes=meshes
            distance=float(distances[i].max())
            frame=self.renderer.frame(self.env,step,distance,bool(success[i]))
            self.cv2.rectangle(frame,(0,0),(960,66),(23,28,36),-1)
            title=f'{self.env.device} PhysX | DEV env{i} {record["region"]} | actor{self.actor_updates} Q{self.critic_updates}'
            status=f'control t={(step+1)/30:.2f}s | max hand distance={distance*100:.1f}cm | success={int(success[i])} unsafe={int(unsafe[i])}'
            self.cv2.putText(frame,title,(15,24),self.cv2.FONT_HERSHEY_SIMPLEX,.50,(240,240,240),1,self.cv2.LINE_AA)
            self.cv2.putText(frame,status,(15,50),self.cv2.FONT_HERSHEY_SIMPLEX,.50,(240,240,240),1,self.cv2.LINE_AA)
            self.writers[i].write(frame);record['frames']+=1;record['last_control_step']=step+1
            self.last_steps[i]=step
            if final:
                self.writers.pop(i).release()
                record['terminal_frame_before_reset']=True
                record['terminal_flags']=dict(success=bool(success[i]),unsafe=bool(unsafe[i]),time_out=bool(time_out[i]))
                self.finalize_video(record,frame)
        self.renderer.meshes=self.meshes

    def finalize_video(self,record,preview_frame=None):
        from browser_video import encode_browser_video
        raw=self.output/record['raw_file']
        if not record['frames'] or record.get('browser_file'):return
        destination=raw.with_name(raw.name.replace('_raw.mp4','_h264.mp4'))
        record['browser_encoding']=encode_browser_video(raw,destination)
        record['browser_file']=destination.name
        if preview_frame is not None:
            preview=destination.with_suffix('.png');self.cv2.imwrite(str(preview),preview_frame)
            record['last_actual_pose_preview']=preview.name
        record['source_video_writer_closed']=True
        # Individual closed media can be attached while other DEV cases run.
        proof=destination.with_suffix('.json');proof.write_text(json.dumps(record,indent=2)+'\n')

    def close(self,outcomes):
        for writer in self.writers.values():writer.release()
        for record in self.records:
            i=record['environment'];record['actual_outcome']=outcomes[i]
            self.finalize_video(record)
            if record.get('browser_file'):
                (self.output/record['browser_file']).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        target=self.output/f'eval_wave_{self.wave:04d}_videos.json'
        target.write_text(json.dumps(dict(records=self.records,physics_not_replayed=True,
            all_recorded_waves_frozen_DEV=True,original_full_denominator_preserved=True),indent=2)+'\n')
        self.writers.clear()
