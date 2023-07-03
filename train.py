import numpy as np
import os
import random
import time

from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from train.dataset import create_replay, fill_replay
from train.peractor import PerceiverIO, PerceiverActorAgent

import argparse
import clip
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(rank, ws, cfg):
    # Set seeds at the beginning of your script
    set_seeds(42)  # very important, in this way you can ensure all the running is consistent and easy to debug

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28766',
                            rank=rank, world_size=ws)
    rank = dist.get_rank()
    print(f"rank = {rank} is initialized")
    torch.cuda.set_device(rank)

    train_replay_storage_dir = '/tmp/replay_train_rank_{}'.format(rank)
    if not os.path.exists(train_replay_storage_dir):
        os.mkdir(train_replay_storage_dir)

    test_replay_storage_dir = '/tmp/replay_test_rank_{}'.format(rank)
    if not os.path.exists(test_replay_storage_dir):
        os.mkdir(test_replay_storage_dir)

    train_replay_buffer = create_replay(cfg, batch_size=cfg['BATCH_SIZE'],
                                        timesteps=1,
                                        save_dir=train_replay_storage_dir,
                                        cameras=cfg['CAMERAS'],
                                        voxel_sizes=cfg['VOXEL_SIZES'])

    test_replay_buffer = create_replay(cfg, batch_size=cfg['BATCH_SIZE'],
                                       timesteps=1,
                                       save_dir=test_replay_storage_dir,
                                       cameras=cfg['CAMERAS'],
                                       voxel_sizes=cfg['VOXEL_SIZES'])

    device = "cuda"
    clip_model, preprocess = clip.load("RN50", device=device)  # CLIP-ResNet50

    def split_demos(total_num_demo, word_size):

        idxs = [i for i in range(total_num_demo)]
        random.shuffle(idxs)
        single_gpu_num_demo = total_num_demo // word_size
        remain_gpu_num_demo = total_num_demo % word_size
        
        if rank < remain_gpu_num_demo:
          # distribute one more demo
          start_index = (single_gpu_num_demo + 1) * rank 
          split_idxs = idxs[start_index:start_index + single_gpu_num_demo + 1]
        else:
          start_index = (single_gpu_num_demo) * rank + remain_gpu_num_demo
          split_idxs = idxs[start_index:start_index + single_gpu_num_demo]
        print('rank',rank,'split_idxs',start_index)

        return split_idxs

    split_idxs = split_demos(cfg['NUM_DEMOS'], ws)
    print("-- Train Buffer --")
    print('rank:{}, demo idxs{}'.format(rank, split_idxs))

    EPISODES_FOLDER = 'train/{}/all_variations/episodes'.format(cfg['TASK'])
    data_path = os.path.join(cfg['DATA_FOLDER'], EPISODES_FOLDER)

    fill_replay(replay=train_replay_buffer,
                start_idx=0,
                demo_idxs=split_idxs,
                data_path=data_path,
                demo_augmentation=True,
                demo_augmentation_every_n=cfg['DEMO_AUGMENTATION_EVERY_N'],
                cameras=cfg['CAMERAS'],
                rlbench_scene_bounds=cfg['SCENE_BOUNDS'],
                voxel_sizes=cfg['VOXEL_SIZES'],
                rotation_resolution=cfg['ROTATION_RESOLUTION'],
                crop_augmentation=False,
                clip_model=clip_model,
                device=device)

    print("-- Test Buffer --")
    fill_replay(replay=test_replay_buffer,
                start_idx=0,
                demo_idxs=list(range(cfg['NUM_DEMOS'], cfg['NUM_DEMOS']+cfg['NUM_TEST'])),
                data_path=data_path,
                demo_augmentation=True,
                demo_augmentation_every_n=cfg['DEMO_AUGMENTATION_EVERY_N'],
                cameras=cfg['CAMERAS'],
                rlbench_scene_bounds=cfg['SCENE_BOUNDS'],
                voxel_sizes=cfg['VOXEL_SIZES'],
                rotation_resolution=cfg['ROTATION_RESOLUTION'],
                crop_augmentation=False,
                clip_model=clip_model,
                device=device)

    # delete the CLIP model since we have already extracted language features
    del clip_model

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
    train_dataset = train_wrapped_replay.dataset()
    train_data_iter = iter(train_dataset)

    # test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    # test_dataset = test_wrapped_replay.dataset()
    # test_data_iter = iter(test_dataset)

    # From https://github.com/stepjam/ARM/blob/main/arm/c2farm/voxel_grid.py

    # initialize PerceiverIO Transformer
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=cfg['VOXEL_SIZES'][0],
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        num_latents=cfg['NUM_LATENTS'],
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
    )

    # initialize PerceiverActor
    peract_agent = PerceiverActorAgent(
        coordinate_bounds=cfg['SCENE_BOUNDS'],
        perceiver_encoder=perceiver_encoder,
        camera_names=cfg['CAMERAS'],
        batch_size=cfg['BATCH_SIZE'],
        voxel_size=cfg['VOXEL_SIZES'][0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=cfg['LR'],
        image_resolution=[cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE']],
        lambda_weight_l2=0.000001,
        transform_augmentation=cfg['TRANSFORM_AUGMENTATION'],
        transform_augmentation_xyz=cfg['TRANSFORM_AUGMENTATION_XYZ'],
        transform_augmentation_rpy=cfg['TRANSFORM_AUGMENTATION_RPY'],
        optimizer_type='lamb',
    )
    peract_agent.build(training=True, device=device, rank=rank)

    save_base = cfg['SAVE_DIR']
    save_dir = cfg['TASK'] + cfg['SAVE_SUFFIX']
    save_dir = os.path.join(save_base, save_dir)
    start_iter = 0
    if os.path.exists(save_dir):
        # resume
        ckpts = os.listdir(save_dir)
        if len(ckpts) > 0:
            ckpts = sorted(ckpts, key=lambda x: int(x))
            last_ckpt = ckpts[-1]
            last_ckpt_path = os.path.join(save_dir, last_ckpt)

            print('resuming from {}'.format(last_ckpt))
            peract_agent.load_weights(last_ckpt_path)
            start_iter = int(last_ckpt)
    # train code

    loss_sum, last_log_iteration = 0, -1
    start_time = time.time()
    for iteration in range(start_iter, cfg['TRAINING_ITERATIONS']):

        batch = next(train_data_iter)
        # print('rank {} batch {}'.format(rank, batch['trans_action_indicies']))
        batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
        update_dict = peract_agent.update(iteration, batch)

        if rank == 0:

            loss_sum = loss_sum + update_dict['total_loss']

            if iteration % cfg['LOG_FREQ'] == 0:
                elapsed_time = (time.time() - start_time) / 60.0
                log_interval = iteration - last_log_iteration
                print("RANK: %d | ITER: %d | Total Loss: %f | Elapsed Time: %f mins" % (
                rank, iteration, loss_sum / log_interval, elapsed_time))
                loss_sum, last_log_iteration = 0, iteration

            if iteration % cfg['SAVE_FREQ'] == 0 and iteration >= cfg['MINIMUM_SAVE_ITER'] or iteration == cfg[
                'TRAINING_ITERATIONS'] - 1:
                weight_dir = os.path.join(save_dir, str(iteration))
                os.makedirs(weight_dir, exist_ok=True)
                peract_agent.save_weights(weight_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' ')

    parser.add_argument('--ws', type=int, help='word size')
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()

    ws = args.ws
    cfg_path = args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件

    mp.spawn(train, nprocs=ws, args=(ws, cfg))
