'''
This script aims at generating the data from collected tasks in the form of behaviours
We want to divide the origin task into behaviours
For each behaviour, we aggregate demos from all tasks
To achieve this goal, first we need to check all the behaviours in the dataset
and then we break down the origin tasks into behaviours
To prevent the model being influenced by the timestamp,
we disable the timestamp in the GPT version of Peract
'''
import json
import os
import numpy as np
import cv2
import csv
import pickle
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation as R

_USING_quaternion = True
try:
    import quaternion
except ImportError:
    quaternion = None
    _USING_quaternion = False


def generate_transformation(rotation, translation=None):
    """
    :param rotation: (N, 3, 3) or (3, 3) -> np.float
    :param translation: (N, 3) or (3,) or None -> np.float
    :return: (N, 4, 4) or (4, 4) -> np.float (return homogeneous transformation matrix)
    """

    transformation = np.identity(4)
    if len(rotation.shape) == 3:
        N = rotation.shape[0]
        transformation = np.expand_dims(transformation, 0)
        transformation = np.tile(transformation, (N, 1, 1))
        transformation[:, :-1, :-1] = rotation
        if translation is not None:
            transformation[:, :-1, -1] = translation
    elif len(rotation.shape) == 2:
        transformation[:-1, :-1] = rotation
        if translation is not None:
            transformation[:-1, -1] = translation
    else:
        raise IOError('Rotation matrix must be in the shape of (N, 3, 3) or (3, 3), got {}'.format(rotation.shape))
    return transformation


def convert_xyzquat_to_transformation(xyzquat):
    """
    Change the xyzquat to the transformation format
    :param xyzquat: (N, 7) or (4,) -> np.float
    :return: (N, 4, 4) or (4, 4) -> np.float (return homogeneous transformation matrix)
    """
    if not _USING_quaternion:
        raise ImportError('Please run [pip install numpy-quaternion] to install quaternion.')

    if isinstance(xyzquat, list):
        xyzquat = np.array(xyzquat)
        if len(xyzquat) != 7:
            raise IOError('XYZ quaternions must be a list with a length of 7, '
                          'got list and length is {}'.format(len(xyzquat)))

    if len(xyzquat.shape) == 1:
        rotation = quaternion.from_float_array(xyzquat[3:])
        rotation = quaternion.as_rotation_matrix(rotation)
        transformation = generate_transformation(rotation, xyzquat[:3])
    elif len(xyzquat.shape) == 2:
        rotation = quaternion.from_float_array(xyzquat[:, 3:])
        rotation = quaternion.as_rotation_matrix(rotation)
        transformation = generate_transformation(rotation, xyzquat[:, :3])
    else:
        raise IOError('XYZ quaternions must be in the shape of (N, 7) or (N,), got {}'.format(xyzquat.shape))

    return transformation


def pose_to_4x4mat(pose):
    '''

    Args:
        pose: a 1x7 array x,y,z,ox,oy,oz,ow

    Returns:

    '''
    pos_x, pos_y, pos_z = pose[:3]
    orien_x, orien_y, orien_z, orien_w = pose[3:]
    translation = np.array([[1, 0, 0, pos_x],
                            [0, 1, 0, pos_y],
                            [0, 0, 1, pos_z],
                            [0, 0, 0, 1]])
    quaternion = np.array([[1 - 2 * (orien_y ** 2 + orien_z ** 2),
                            2 * (orien_x * orien_y - orien_z * orien_w),
                            2 * (orien_x * orien_z + orien_y * orien_w), 0],
                           [2 * (orien_x * orien_y + orien_z * orien_w),
                            1 - 2 * (orien_x ** 2 + orien_z ** 2),
                            2 * (orien_y * orien_z - orien_x * orien_w), 0],
                           [2 * (orien_x * orien_z - orien_y * orien_w),
                            2 * (orien_y * orien_z + orien_x * orien_w),
                            1 - 2 * (orien_x ** 2 + orien_y ** 2), 0],
                           [0, 0, 0, 1]])
    return np.matmul(translation, quaternion)


def mat4x4_to_pose(mat):
    '''
    Args:
        mat: a 4x4 transformation matrix

    Returns:
        pose: a 1x7 array x,y,z,ox,oy,oz,ow
    '''
    pos_x, pos_y, pos_z = mat[:3, 3]
    rotation_matrix = mat[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    orien_x, orien_y, orien_z, orien_w = rotation.as_quat()
    pose = np.array([pos_x, pos_y, pos_z, orien_x, orien_y, orien_z, orien_w])
    return pose


def convert_transformation_to_xyzquat(transform):
    """
    Convert the transformation to the xyz+quaternion format
    :param transform: (N, 4, 4) or (4, 4) -> np.float
    :return: (N, 7) or (7,) -> np.float (return imaginary in the behind format (a + bi + cj + dk))
    """
    if not _USING_quaternion:
        raise ImportError('Please run [pip install numpy-quaternion] to install quaternion.')

    if len(transform.shape) == 3:
        quaternions = quaternion.from_rotation_matrix(transform[:, :-1, :-1])
        quaternions = quaternion.as_float_array(quaternions)
        T_Q = np.append(transform[:, :-1, -1], quaternions, axis=1)
    elif len(transform.shape) == 2:
        quaternions = quaternion.from_rotation_matrix(transform[:-1, :-1])
        quaternions = quaternion.as_float_array(quaternions)
        T_Q = np.append(transform[:-1, -1], quaternions)
    else:
        raise IOError(
            'Transformation matrix must be in the shape of (N, 4, 4) or (4, 4), got {}'.format(transform.shape))

    return T_Q


def harvest_pose_from_file(pose_path, gripper_offset=[-0.012, -0.085, 0.360]):
    with open(pose_path) as f:
        data = json.load(f)
        pose = data[0]
    pose = np.array(pose)
    # roll from ow ox oy oz  ->  ox oy oz ow
    pose[3:] = np.roll(pose[3:], -1)

    pose_matrix = pose_to_4x4mat(pose)

    compare_pose_matrix = pose_to_4x4mat(pose)

    # apply gripper offset
    transformation_gripper_offset = np.eye(4)
    transformation_gripper_offset[:, 3][:3] = gripper_offset

    transformation_pose = pose_matrix @ transformation_gripper_offset

    transformation_7d = mat4x4_to_pose(transformation_pose)

    return transformation_7d


def get_obs(misc, pos, suction_state):
    # todo: update the observation according to the real positions and camera arguments
    # finger_positions = np.array(frame['joint_states'].position)[-2:]
    # gripper_open_amount = finger_positions[0] + finger_positions[1]

    # gripper_pose = np.array([
    #     frame['gripper_pose'].pose.position.x,
    #     frame['gripper_pose'].pose.position.y,
    #     frame['gripper_pose'].pose.position.z,
    #     frame['gripper_pose'].orien_x,
    #     frame['gripper_pose'].orien_y,
    #     frame['gripper_pose'].orien_z,
    #     frame['gripper_pose'].orien_w,
    # ])
    gripper_pose = harvest_pose_from_file(pos)  # now the pose is the falan's pose

    obs = Observation(
        left_shoulder_rgb=None,
        left_shoulder_depth=None,
        left_shoulder_point_cloud=None,
        right_shoulder_rgb=None,
        right_shoulder_depth=None,
        right_shoulder_point_cloud=None,
        overhead_rgb=None,
        overhead_depth=None,
        overhead_point_cloud=None,
        wrist_rgb=None,
        wrist_depth=None,
        wrist_point_cloud=None,
        front_rgb=None,
        front_depth=None,
        front_point_cloud=None,
        left_shoulder_mask=None,
        right_shoulder_mask=None,
        overhead_mask=None,
        wrist_mask=None,
        front_mask=None,
        joint_velocities=np.zeros(7),
        joint_positions=gripper_pose,
        joint_forces=np.zeros(7),
        gripper_open=suction_state,
        gripper_pose=gripper_pose,  # todo: update it to the right position
        gripper_matrix=pose_to_4x4mat(gripper_pose),  # todo: update it to the right pose matrix
        gripper_touch_forces=None,
        gripper_joint_positions=np.zeros(2),
        task_low_dim_state=None,
        ignore_collisions=True,  # TODO: fix
        misc=misc,
    )
    return obs


def save_keypoint_gpt(rgb_images, depth_images, poses, cfg, img_size=(128, 128)):
    assert img_size == (128, 128), "do  not support other image size due to it will cause camera intri parameter error"

    # make directories
    def check_and_mkdirs(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    images_num = len(rgb_images)
    assert len(rgb_images) == len(depth_images) and len(rgb_images) == len(poses) and len(rgb_images) > 0 \
           and len(rgb_images) == len(cfg['suction_states'])

    save_path = os.path.join(cfg['demo']['save_path'], cfg['demo']['task'])
    episode_idx = cfg['demo']['episode']
    variation_idx = cfg['demo']['variation']

    episode_path = os.path.join(save_path, 'all_variations', 'episodes', f"episode{episode_idx}")
    check_and_mkdirs(episode_path)

    front_rgb_path = os.path.join(episode_path, 'front_rgb')
    check_and_mkdirs(front_rgb_path)
    front_depth_path = os.path.join(episode_path, 'front_depth')
    check_and_mkdirs(front_depth_path)

    # misc (camera_info etc)
    misc = dict()
    misc['front_camera_intrinsics'] = cfg['front_camera_intrinsics']  # to do
    misc['front_camera_extrinsics'] = cfg['front_camera_extrinsics']
    misc['front_camera_near'] = 0.5  # to modify
    misc['front_camera_far'] = 4.5  # to modify

    misc['keypoint_idxs'] = np.arange(images_num)

    observations = []
    jumped_cnt = 0
    for f_idx in range(images_num):
        suction_state = cfg['suction_states'][f_idx]
        if suction_state == -1:
            jumped_cnt += 1
            continue
        save_idx = f_idx

        front_rgb = rgb_images[f_idx]
        front_depth = depth_images[f_idx]

        # copy and rename the images
        front_rgb_filename = os.path.join(front_rgb_path, f'{f_idx - jumped_cnt}.png')
        rgb_image = cv2.imread(front_rgb)
        rgb_image = cv2.resize(rgb_image, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(front_rgb_filename, rgb_image)

        front_depth_filename = os.path.join(front_depth_path, f'{f_idx - jumped_cnt}.png')
        depth_image = translate_8UC4_to_16UC1(front_depth)
        depth_image = depth_image.astype(np.float32)
        depth_image = cv2.resize(depth_image, img_size, interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        cv2.imwrite(front_depth_filename, depth_image)

        # pose used in here

        pose = poses[f_idx]

        # check time
        rgb_time = os.path.getctime(front_rgb)
        depth_time = os.path.getctime(front_depth)
        pose_time = os.path.getctime(pose)

        if not (abs(rgb_time - depth_time) < 2.1 and abs(depth_time - pose_time) < 2.1):
            print(rgb_time, depth_time, pose_time)
            print(front_rgb, front_depth, pose)

        observations.append(get_obs(misc, pose, suction_state))

    demo = Demo(observations, random_seed=0)
    demo.variation_number = variation_idx

    low_dim_obs_path = os.path.join(episode_path, 'low_dim_obs.pkl')
    with open(low_dim_obs_path, 'wb') as f:
        pickle.dump(demo, f)

    variation_number_path = os.path.join(episode_path, 'variation_number.pkl')
    with open(variation_number_path, 'wb') as f:
        pickle.dump(variation_idx, f)

    descriptions = cfg['bhvs']
    descriptions_path = os.path.join(episode_path, 'variation_descriptions.pkl')
    with open(descriptions_path, 'wb') as f:
        pickle.dump(descriptions, f)

    print(f"Saved {images_num} frames to {save_path}")


def generate_cfg_gpt(save_path, variation_idx, episode_idx, task, bhvs, suction_states) -> dict:
    cfg = {}
    cfg['demo'] = {}
    cfg['demo']['save_path'] = os.path.join(save_path, 'behaviours')
    cfg['demo']['task'] = task
    cfg['demo']['episode'] = episode_idx
    cfg['demo']['variation'] = variation_idx
    # cfg['demo']['lang_goal'] = lang_goal
    cfg['front_camera_intrinsics'] = np.array([[182.29981142, 0., 64.4480685],
                                               [0., 291.64949166, 66.78173778],
                                               [0., 0., 1.],
                                               ])
    front_camera_extrinsics = np.array([0.1087458851426165,
                                        -1.0974391229818714,
                                        0.5189316107518424,
                                        -0.336851733424418,
                                        0.9396495772780166,
                                        0.023653515121718285,
                                        -0.055046279007725026])
    front_camera_extrinsics[3:] = np.roll(front_camera_extrinsics[3:], -1)  # roll
    cfg['front_camera_extrinsics'] = pose_to_4x4mat(front_camera_extrinsics)
    cfg['suction_states'] = suction_states
    bhvs.append('end')
    cfg['bhvs'] = bhvs
    return cfg


def translate_8UC4_to_16UC1(depth):
    """
    Convert depth image from 8UC4 format to 16UC1.
    :param depth: (H, W, 4) -> np.float
    :return: depth in 16UC1 format

    `8UC4`: depth in mm
    `16UC1`: depth in deciMm
    `32FC1`: target format to train, in mm

    """

    if isinstance(depth, str):
        depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    if len(depth.shape) == 3 and depth.shape[2] == 4 and depth.dtype == np.uint8:
        dep = depth.view(dtype=np.float32)
        dep = dep.astype(dtype=np.uint16)
        return dep
    elif len(depth.shape) == 2 and depth.dtype == np.uint16:
        return depth
    else:
        raise TypeError('Invalid Depth Image, got shape {}'.format(depth.shape))

def get_data(data_folder_path):
    ##############ensure that you check the above info
    two_dirs = os.listdir(data_folder_path)
    assert len(two_dirs) == 2
    time_dir_name = None
    for dir_name in two_dirs:
        if dir_name.startswith('2023'):
            time_dir_name = dir_name
    assert time_dir_name
    rgb_images_path = os.path.join(data_folder_path, time_dir_name, 'KEC10215A3000626-color')
    depth_images_path = os.path.join(data_folder_path, time_dir_name, 'KEC10215A3000626-depth')
    poses_path = os.path.join(data_folder_path, 'falan')
    ####################################################

    rgb_images = os.listdir(rgb_images_path)
    rgb_images.sort()
    rgb_images = [os.path.join(rgb_images_path, img_name) for img_name in rgb_images]

    depth_images = os.listdir(depth_images_path)
    depth_images.sort()
    depth_images = [os.path.join(depth_images_path, img_name) for img_name in depth_images]

    poses = os.listdir(poses_path)
    poses.sort()
    poses = [os.path.join(poses_path, pose) for pose in poses]

    return rgb_images, depth_images, poses

task_suction_states = {
    # 'put_in': [0, 1, 1, -1, 0],
    'put_in': [0, 1, 1, 0],
    'stack': [0, 1, 1, 1, 0],
    'move': [0, 1, 1, 1, 0],
    'cover': [0, 1, 1, 1, 0],
    'press': [1, 1, 1],
    'hit': [0, 1, 1],
    'rotate': [0, 1, 1, 1, 0],
    'bring': [0, 1, 0],
    'push': [1, 1, 1, 1, 1, 1],
    'uncover': [0, 1, 1, 1, 0, 0, 1, 1],
    'valid': [-1],
    'close': [0, 1, 1, 1, 0]
}

behaviours = {
    0: 'get poker',
    1: 'get nuts',
    2: 'get oil',
    3: 'get cream',
    4: 'get bowl',
    5: 'get cover',
    6: 'get yellow block',
    7: 'get green block',
    8: 'get tool',
    9: 'pull up',
    10: 'drop box',
    11: 'drop bowl',
    12: 'drop left',
    13: 'drop right',
    14: 'drop down',
    15: 'drop pink block',
    16: 'drop yellow block',
    17: 'drop cream',
    18: 'drop oil',
    19: 'drop cream',
    20: 'drop cover on cream',
    21: 'move left',
    22: 'move right',
    23: 'move back',
    24: 'move blue block',
    25: 'move pink block',
    26: 'move poker',
    27: 'move oil',
    28: 'above pink block',
    29: 'above yellow block',
    30: 'above oil',
    31: 'above cream',
    32: 'above button',
    33: 'behind blue block',
    34: 'right of blue block',
    35: 'rotate',
    36: 'press',
    37: 'align',
    38: 'push left',
    39: 'push right'
}

prompt_behaviours = {
    'put the poker in the box': [0, 9, 10],
    'put the poker in the bowl': [0, 9, 11],
    'put the nuts in the box': [1, 9, 10],
    'put the nuts in the bowl': [1, 9, 11],
    'move the yellow block to the left': [6, 9, 21, 14],
    'move the yellow block to the right': [6, 9, 22, 14],
    'stack the green block on the pink block': [7, 9, 28, 15],
    'stack the green block on the yellow block': [7, 9, 29, 16],
    'cover the oil': [4, 9, 30, 18],
    'cover the cream': [4, 9, 31, 19],
    'hit the blue block': [8, 24],
    'hit the pink block': [8, 25],
    'rotate the oil': [2, 9, 35, 14],
    'rotate the poker': [0, 9, 35, 14],
    'bring the cream to the poker': [3, 26],
    'bring the nuts to the oil': [2, 27],
    'press the button': [32, 36],
    'push the blue block to the left': [33, 37, 9, 34, 38],
    'push the blue block to the right': [33, 37, 9, 34, 39],
    'close the cream': [5, 9, 20, 35],
    'uncover the bowl and pick up the oil': [4, 9, 23, 14, 9, 2, 9],
    'uncover the bowl and pick up the cream': [4, 9, 23, 14, 9, 3, 9]
}

if __name__ == '__main__':
    # this script is for data collection
    print('start transfer data to required format')
    ####required path
    base_path = r'D:\datasets\\real_world\\valid'
    task_name = os.path.basename(base_path)
    assert task_name in task_suction_states or task_name == 'valid', 'the task did not exist in suction states dict'
    suction_states = task_suction_states[task_name]

    with open('data_utils/csv/{}.csv'.format(task_name), 'r') as f:

        reader = csv.reader(f)
        header = reader.__next__()
        for line in reader:

            print(line)
            name, task, lang_goal, episode_idx, variation_idx = line
            prompt = lang_goal.strip()
            assert prompt in prompt_behaviours, 'prompt {} not in dict'.format(prompt)
            assert task in task_suction_states, task
            suction_states = task_suction_states[task]

            # get valid suction states
            if task_name == 'valid':
                # update the suctions states according to the folder name
                flag = 0
                for t in task_suction_states.keys():
                    if t in name:
                        suction_states = task_suction_states[t]
                        flag += 1
                if 'uncover' in name:
                    suction_states = task_suction_states['uncover']
                    flag = 1
                assert flag == 1

            bhvs = prompt_behaviours[prompt]
            episode_idx, variation_idx = int(episode_idx), int(variation_idx)

            data_folder_path = os.path.join(base_path, name)
            assert os.path.exists(data_folder_path)

            rgb_images, depth_images, poses = get_data(data_folder_path)


            # log bhvs
            for bhv_id in bhvs:
                # behaviour
                print(behaviours[bhv_id])

            jumped_cnt = 0
            for ss in suction_states:
                if ss == -1:
                    # jump
                    jumped_cnt += 1
            assert len(bhvs) == len(suction_states) - jumped_cnt - 1, 'number of bhvs not match suction_states'
            bhvs = [behaviours[bhv_id] for bhv_id in bhvs]

            cfg = generate_cfg_gpt(base_path, variation_idx, episode_idx, task, bhvs, suction_states)

            episode_path = os.path.join(base_path,'behaviours', task, 'all_variations', 'episodes', f"episode{episode_idx}")
            if os.path.exists(episode_path):
                print('exist jump')
                continue




            save_keypoint_gpt(rgb_images, depth_images, poses, cfg)
