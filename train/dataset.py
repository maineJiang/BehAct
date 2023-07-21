import numpy as np
import os
import sys
import pickle
from rlbench.utils import get_stored_demo_front
from rlbench.backend.utils import extract_obs

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer import ReplayElement
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from rlbench.demo import Demo
from typing import List

import clip
import arm.utils as utils

from rlbench.backend.observation import Observation
from yarr.replay_buffer.replay_buffer import ReplayBuffer
import torch
import torch.distributed as dist

# settings
EPISODE_FOLDER = 'episode%d'  # PerceiverIO latents  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
VARIATION_DESCRIPTIONS_PKL = 'variation_descriptions.pkl'  # the pkl file that contains language goals for each demonstration
def create_replay(cfg, batch_size: int,
                  timesteps: int,
                  save_dir: str,
                  cameras: list,
                  voxel_sizes,
                  replay_size=3e5):
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (cfg['LOW_DIM_SIZE'],), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'],), np.float32))
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'],), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'],),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),
        ReplayElement('lang_goal_embs', (max_token_seq_len, lang_emb_dim,),  # extracted from CLIP's language encoder
                      np.float32),
        ReplayElement('lang_goal', (1,), object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool_),
    ]

    replay_buffer = UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def _keypoint_discovery(demo: Demo,
                        stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                       last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _all_keypoint(demo: Demo,
                  stopping_delta=0.1) -> List[int]:
    l = len(demo)
    episode_keypoints = list(range(l))
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb


# add individual data points to replay
def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '', # or list
        clip_model=None,
        device='cpu'):
    flag_bhvs = type(description) == list
    prev_action = None
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes,
            rotation_resolution, crop_augmentation)
        # print(trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * 1.0 if terminal else 0

        if flag_bhvs:
          desc = description[keypoint] 
          obs_dict = extract_obs(obs, cameras, t=0, prev_action=prev_action) #ignore timestamp
        else:
          desc = description
          obs_dict = extract_obs(obs, cameras, t=k, prev_action=prev_action)
        tokens = clip.tokenize([desc]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'lang_goal': np.array([desc], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    if flag_bhvs:
      obs_dict_tp1 = extract_obs(obs_tp1, cameras, t=0, prev_action=prev_action) #ignore timestamp
    else:
      obs_dict_tp1 = extract_obs(obs_tp1, cameras, t=k + 1, prev_action=prev_action)
    obs_dict_tp1['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(replay: ReplayBuffer,
                start_idx: int,
                demo_idxs: List[int],
                data_path: str,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model=None,
                device='cpu'):
    print('Filling replay ...')
    rank = dist.get_rank()
    for d_idx in demo_idxs:
        # print("rank %d Filling demo %d" % (rank, d_idx))
        # print('rank {} device {}'.format(rank, device))
        demo = get_stored_demo_front(data_path=data_path,
                                     index=d_idx)

        # get language goal from disk
        varation_descs_pkl_file = os.path.join(data_path, EPISODE_FOLDER % d_idx, VARIATION_DESCRIPTIONS_PKL)
        with open(varation_descs_pkl_file, 'rb') as f:
            descs = pickle.load(f)
        assert len(descs) == 1 or len(descs) == len(demo)
        flag_bhvs = len(descs) == len(demo)
        
        # extract keypoints
        episode_keypoints = _all_keypoint(demo)  # all is keypoint
        
        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                continue

            obs = demo[i]
            if flag_bhvs:
              desc = descs
            else:
              desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            _add_keypoints_to_replay(
                replay, obs, demo, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device)
    print('Replay filled with demos. {} in total'.format(len(demo_idxs)))

