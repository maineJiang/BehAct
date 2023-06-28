import argparse
import numpy as np
import os
import random
import time

import clip
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from myyarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from train.dataset import create_replay, fill_replay
from train.peractor import PerceiverIO, PerceiverActorAgent
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def valid(rank, ws, cfg):
    # valid on single gpu
    set_seeds(42)

    assert ws == 1
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765',
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
        start_index = single_gpu_num_demo * rank
        split_idxs = idxs[start_index:start_index + single_gpu_num_demo]

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


    test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    test_dataset = test_wrapped_replay.dataset()
    test_data_iter = iter(test_dataset)

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
        lr=0.0001,
        image_resolution=[cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE']],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )
    peract_agent.build(training=False, device=device, rank=rank)

    save_base = cfg['SAVE_DIR']
    save_dir = cfg['TASK'] + cfg['SAVE_SUFFIX']
    save_dir = os.path.join(save_base, save_dir)
    weight_dir = os.path.join(save_dir, str(cfg['WEIGHT']))
    peract_agent.load_weights(weight_dir)


    # train code

    loss_sum, last_log_iteration = 0, -1
    small_diff_number, big_diff_number = 0, 0
    start_time = time.time()

    # with torch.no_grad():
    for iteration in range(cfg['TESTING_ITERATIONS']):


            batch = next(test_data_iter)
            batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
            update_dict = peract_agent.update(iteration, batch)

            if rank == 0:
                vis_trans_coord = update_dict['pred_action']['trans'][0].detach().cpu().numpy()
                vis_gt_coord = update_dict['expert_action']['action_trans'][0].detach().cpu().numpy()
                loss = update_dict['total_loss']
                loss_sum = loss_sum + loss

                diff = np.abs(vis_trans_coord - vis_gt_coord).sum()
                if diff < 10:
                    print('diff is small', 'vis_trans_coord', vis_trans_coord, 'vis_gt_coord', vis_gt_coord)
                    small_diff_number = small_diff_number + 1
                else:
                    print('diff is big', 'vis_trans_coord', vis_trans_coord, 'vis_gt_coord', vis_gt_coord)
                    big_diff_number = big_diff_number + 1

                if iteration % 1 == 0:
                    elapsed_time = (time.time() - start_time) / 60.0
                    log_interval = iteration - last_log_iteration
                    print("ITER: %d | Total Loss: %f | Elapsed Time: %f mins" % (iteration, loss_sum / log_interval, elapsed_time))
                    loss_sum, last_log_iteration = 0, iteration

    print('small_diff_number: ', small_diff_number, 'big_diff_number: ', big_diff_number)

    dist.destroy_process_group()

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description=' ')

    parser.add_argument('--ws', type=int, help='word size')
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()

    ws = args.ws
    cfg_path = args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    mp.spawn(valid, nprocs=ws, args=(ws, cfg))
