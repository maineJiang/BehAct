import copy
import os
from functools import reduce as funtool_reduce
from functools import wraps
from operator import mul

import cv2
import clip
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from arm.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock
from arm.optim.lamb import Lamb
from arm.vision_sensor import VisionSensor
from arm.utils import discrete_euler_to_quaternion
from train.show_points_inference import visualize_from_prediction

# settings
VOXEL_SIZES = [100]  # 100x100x100 voxels
NUM_LATENTS = 512  # PerceiverIO latents
# SCENE_BOUNDS = [-0.385, -0.669, -0.2, 0.314, -0.223,
#                 0.2]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS = [-0.385, -0.711, -0.21, 0.314, -0.179,
                0.09]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
BATCH_SIZE = 1
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}
IMAGE_SIZE = 128  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate
# the dataset with larger images
VARIATION_DESCRIPTIONS_PKL = 'variation_descriptions.pkl'  # the pkl file that contains language goals for each demonstration
EPISODE_LENGTH = 10  # max steps for agents
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
CAMERAS = ['front']


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(replay_sample):
    obs, pcds = [], []
    for n in CAMERAS:
        rgb = stack_on_channel(replay_sample['%s_rgb' % n])
        pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])

        rgb = _norm_rgb(rgb)

        obs.append([rgb, pcd])  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


# PerceiverIO adapted for 6-DoF manipulation
# From: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,
                 max_num_coords: int, ):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              device=device).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          device=device).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], device=device), max_dims,
             torch.tensor([4 + feature_size], device=device)], -1).tolist()
        self._ones_max_coords = torch.ones((batch_size, max_num_coords, 1),
                                           device=device)
        self._num_coords = max_num_coords

        shape = self._total_dims_list

        self._result_dim_sizes = torch.tensor(
            [funtool_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [
                1], device=device)
        flat_result_size = funtool_reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float,
                                         device=device)
        self._flat_output = torch.ones(flat_result_size, dtype=torch.float,
                                       device=device) * self._initial_val
        self._arange_to_max_coords = torch.arange(4 + feature_size,
                                                  device=device)
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float,
                                       device=device)

        self._const_1 = torch.tensor(1.0, device=device)
        self._batch_size = batch_size

        # Coordinate Bounds:
        self._bb_mins = self._coord_bounds[..., 0:3]
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - self._bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        self._dims_orig = dims_orig = self._voxel_shape_spec.int() - 2
        self._dims_m_one = (dims - 1).int()
        # BS x 1 x 3
        self._res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)

        self._voxel_indicy_denmominator = self._res + MIN_DENOMINATOR
        self._dims_m_one_zeros = torch.zeros_like(self._dims_m_one)

        batch_indices = torch.arange(self._batch_size, dtype=torch.int,
                                     device=device).view(self._batch_size, 1, 1)
        self._tiled_batch_indices = batch_indices.repeat(
            [1, self._num_coords, 1])

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, device=device)
        self._index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None,
                                      coord_bounds=None):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # global-coordinate point cloud (x, y, z)
        voxel_values = coords

        # rgb values (R, G, B)
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)  # concat rgb values (B, 128, 128, 3)

        # coordinates to aggregate over
        _, num_coords, _ = voxel_indices.shape
        all_indices = torch.cat([
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)

        # max coordinates
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # aggregate across camera views
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size))

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        # occupied value
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)

        # hard voxel-location position encoding
        return torch.cat(
            [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
             vox[..., -1:]], -1)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):  # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxel_grid: VoxelGrid,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxel_grid = voxel_grid
        self._qnet = copy.deepcopy(perceiver_encoder)
        self._qnet._dev = device

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                obs,
                proprio,
                pcd,
                lang_goal_embs,
                bounds=None):

        # flatten point cloud
        bs = obs[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)

        # flatten rgb
        image_features = [o[0] for o in obs]
        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in
             image_features], 1)

        # voxelize
        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat.float(), coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # visualize
        '''from arm.utils import visualise_voxel
        import matplotlib.pyplot as plt
        rotation_amount = -90  # @param {type:"slider", min:-180, max:180, step:5}
        rendered_img = visualise_voxel(voxel_grid[0].cpu().numpy(),
                                       None,
                                       None,
                                       [0, 0, 0],
                                       voxel_size=0.045,
                                       rotation_amount=np.deg2rad(rotation_amount))

        fig = plt.figure(figsize=(15, 15))
        plt.imshow(rendered_img)
        plt.axis('off')
        plt.savefig('inference_voxel_image_train.png')
        plt.close()'''

        # batch bounds if necessary
        if bounds.shape[0] != bs:
            bounds = bounds.repeat(bs, 1)

        # forward pass
        q_trans, rot_and_grip_q, collision_q = self._qnet(voxel_grid,
                                                          proprio,
                                                          lang_goal_embs,
                                                          bounds)
        return q_trans, rot_and_grip_q, collision_q, voxel_grid

    def latents(self):
        return self._qnet.latent_dict


class PerceiverIO(nn.Module):
    def __init__(
            self,
            depth,  # number of self-attention layers
            iterations,  # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,  # N voxels per side (size: N*N*N)
            initial_dim,  # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,
            # 4 dimensions - proprioception: {gripper_open, left_finger_joint, right_finger_joint, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,  # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,  # 3D tensors have 3 axes
            num_latents=512,  # number of latent vectors
            im_channels=64,  # intermediate channel size
            latent_dim=512,  # dimensions of latent vectors
            cross_heads=1,  # number of cross-attention heads
            latent_heads=8,  # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            voxel_patch_size=5,  # intial patch size
            voxel_patch_stride=5,  # initial stride to patchify voxel input
            final_dim=64,  # final dimensions of features
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features
        self.input_dim_before_seq = self.im_channels * 2

        # learnable positional encoding
        lang_emb_dim, lang_max_seq_len = 512, 77
        self.pos_encoding = nn.Parameter(torch.randn(1,
                                                     lang_max_seq_len + spatial_size ** 3,
                                                     self.input_dim_before_seq))

        # voxel input preprocessing encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            self.low_dim_size, self.im_channels, norm=None, activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # lang preprocess
        self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 2)

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size,
            self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, self.input_dim_before_seq, heads=cross_heads,
                                          dim_head=cross_dim_head, dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads,
                                                    dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq,
                                          Attention(self.input_dim_before_seq, latent_dim, heads=cross_heads,
                                                    dim_head=cross_dim_head,
                                                    dropout=decoder_dropout),
                                          context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final layers
        self.final = Conv3DBlock(
            self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        # 100x100x100x64 -> 100x100x100x1 decoder for translation Q-values
        self.trans_decoder = Conv3DBlock(
            self.final_dim, 1, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        # final 3D softmax
        self.ss_final = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size,
            self.im_channels)

        flat_size += self.im_channels * 4

        # MLP layers
        self.dense0 = DenseBlock(
            flat_size, 256, None, activation)
        self.dense1 = DenseBlock(
            256, self.final_dim, None, activation)

        # 1x64 -> 1x(72+72+72+2+2) decoders for rotation, gripper open, and collision Q-values
        self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                self.num_rotation_classes * 3 + \
                                                self.num_grip_classes + \
                                                self.num_collision_classes,
                                                None, None)

    def forward(
            self,
            ins,
            proprio,
            lang_goal_embs,
            bounds,
            mask=None,
    ):
        # preprocess
        d0 = self.input_preprocess(ins)  # [B,10,100,100,100] -> [B,64,100,100,100]

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)  # [B,64,100,100,100] -> [B,64,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio
        p = self.proprio_preprocess(proprio.float())  # [B,4] -> [B,64]
        p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
        ins = torch.cat([ins, p], dim=1)  # [B,128,20,20,20]

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')  # [B,20,20,20,128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten voxel grid into sequence
        ins = rearrange(ins, 'b ... d -> b (...) d')  # [B,8000,128]

        # append language features as sequence
        l = self.lang_preprocess(lang_goal_embs)  # [B,77,1024] -> [B,77,128]
        ins = torch.cat((l, ins), dim=1)  # [B,8077,128]

        # add learable pos encoding
        ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins, context=x)
        latents = latents[:, l.shape[1]:]

        # reshape back to voxel grid
        latents = latents.view(b, *ins_orig_shape[1:-1], latents.shape[-1])  # [B,20,20,20,64]
        latents = rearrange(latents, 'b ... d -> b d ...')  # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders
        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample layer
        u0 = self.up0(latents)  # [B,64,100,100,100]

        # skip connection like in UNets
        u = self.final(torch.cat([d0, u0], dim=1))  # [B,64+64,100,100,100] -> [B,64,100,100,100]

        # translation decoder
        trans = self.trans_decoder(u)  # [B,64,100,100,100] -> [B,1,100,100,100]

        # aggregated features from final softmax and maxpool for MLP decoders
        feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])

        # decoder MLP layers for rotation, gripper open, and collision
        dense0 = self.dense0(torch.cat(feats, dim=1))
        dense1 = self.dense1(dense0)  # [B,72*3+2+2]

        # format output
        rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
        rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
        collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        return trans, rot_and_grip_out, collision_out


class PerceiverActorAgent():
    def __init__(self,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 voxel_feature_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'lamb'):

        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = transform_augmentation_xyz
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        # suction force
        self._suction_open = 0

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
        )
        self._vox_grid = vox_grid

        self._q = QFunction(self._perceiver_encoder,
                            vox_grid,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._optimizer_type == 'lamb':
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == 'adam':
            self._optimizer = torch.optim.Adam(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception('Unknown optimizer')

    def _softmax_q(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _get_one_hot_expert_actions(self,
                                    # You don't really need this function since GT labels are already in the right format. This is some leftover code from my experiments with label smoothing.
                                    batch_size,
                                    action_trans,
                                    action_rot_grip,
                                    action_ignore_collisions,
                                    device):
        bs = batch_size

        # initialize with zero tensors
        action_trans_one_hot = torch.zeros((bs, self._voxel_size, self._voxel_size, self._voxel_size), dtype=int,
                                           device=device)
        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            # translation
            gt_coord = action_trans[b, :]
            action_trans_one_hot[b, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

            # rotation
            gt_rot_grip = action_rot_grip[b, :]
            action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
            action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
            action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
            action_grip_one_hot[b, gt_rot_grip[3]] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        # flatten trans
        action_trans_one_hot = action_trans_one_hot.view(bs, -1)

        return action_trans_one_hot, \
               action_rot_x_one_hot, \
               action_rot_y_one_hot, \
               action_rot_z_one_hot, \
               action_grip_one_hot, \
               action_collision_one_hot

    def update(self, step: int, replay_sample: dict, backprop: bool = True) -> dict:
        # sample
        action_trans = replay_sample['trans_action_indicies'][:, -1, :3].int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1].int()
        action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_gripper_pose = replay_sample['gripper_pose'][:, -1]
        lang_goal_embs = replay_sample['lang_goal_embs'][:, -1].float()

        # metric scene bounds
        bounds = bounds_tp1 = self._coordinate_bounds

        # inputs
        proprio = stack_on_channel(replay_sample['low_dim_state'])
        obs, pcd = _preprocess_inputs(replay_sample)

        # TODO: data augmentation by applying SE(3) pertubations to pcd and actions
        # see https://github.com/peract/peract/blob/main/voxel/augmentation.py#L68 for reference

        # Q function
        q_trans, rot_grip_q, collision_q, voxel_grid = self._q(obs,
                                                               proprio,
                                                               pcd,
                                                               lang_goal_embs,
                                                               bounds)

        # one-hot expert actions
        bs = self._batch_size
        action_trans_one_hot, action_rot_x_one_hot, \
        action_rot_y_one_hot, action_rot_z_one_hot, \
        action_grip_one_hot, action_collision_one_hot = self._get_one_hot_expert_actions(bs,
                                                                                         action_trans,
                                                                                         action_rot_grip,
                                                                                         action_ignore_collisions,
                                                                                         device=self._device)
        total_loss = 0.
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans.view(bs, -1),
                                                  action_trans_one_hot.argmax(-1))

            rot_grip_loss = 0.
            rot_grip_loss += self._cross_entropy_loss(
                rot_grip_q[:, 0 * self._num_rotation_classes:1 * self._num_rotation_classes],
                action_rot_x_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(
                rot_grip_q[:, 1 * self._num_rotation_classes:2 * self._num_rotation_classes],
                action_rot_y_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(
                rot_grip_q[:, 2 * self._num_rotation_classes:3 * self._num_rotation_classes],
                action_rot_z_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 3 * self._num_rotation_classes:],
                                                      action_grip_one_hot.argmax(-1))

            collision_loss = self._cross_entropy_loss(collision_q,
                                                      action_collision_one_hot.argmax(-1))

            total_loss = trans_loss + rot_grip_loss + collision_loss
            total_loss = total_loss.mean()

            # backprop
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            total_loss = total_loss.item()

        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)

        # discrete to continuous translation action
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2

        return {
            'total_loss': total_loss,
            'voxel_grid': voxel_grid,
            'q_trans': self._softmax_q(q_trans),
            'pred_action': {
                'trans': coords_indicies,
                'continuous_trans': continuous_trans,
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }

    def act(self, replay_sample: dict) -> dict:
        # inputs
        proprio = stack_on_channel(replay_sample['low_dim_state'])
        # update the suction force
        proprio[0, 1] = self._suction_open

        obs, pcd = _preprocess_inputs(replay_sample)
        lang_goal_embs = replay_sample['lang_goal_embs'].float()

        # Q function
        q_trans, rot_grip_q, collision_q, voxel_grid = self._q(obs,
                                                               proprio,
                                                               pcd,
                                                               lang_goal_embs,
                                                               self._coordinate_bounds)

        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)

        # discrete to continuous translation action
        bounds = self._coordinate_bounds
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2

        # update the suction state according to prediction
        prediction_suction = rot_and_grip_indicies[0, 3].item()
        self._suction_open = prediction_suction

        return {
            'voxel_grid': voxel_grid,
            'q_trans': self._softmax_q(q_trans),
            'pred_action': {
                'trans': coords_indicies,
                'continuous_trans': continuous_trans,
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },

        }

    def load_weights(self, savedir: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_file = os.path.join(savedir, '%s.pt' % 'peract')
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('module._qnet', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    print("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % 'peract'))


def init_perceiver_actor(perceiver_actor_weight_path):
    # initialize PerceiverIO Transformer
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=VOXEL_SIZES[0],
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        num_latents=NUM_LATENTS,
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

    assert torch.cuda.is_available(), "the model need to init on a available gpu"
    peract_agent = PerceiverActorAgent(
        coordinate_bounds=SCENE_BOUNDS,
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=BATCH_SIZE,
        voxel_size=VOXEL_SIZES[0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )
    peract_agent.build(training=False, device='cuda:0')
    peract_agent.load_weights(perceiver_actor_weight_path)

    return peract_agent


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


def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

  Reverses the depth to image conversion performed by FloatArrayToRgbImage or
  FloatArrayToGrayImage.

  The image is treated as an array of fixed point depth values.  Each
  value is converted to float and scaled by the inverse of the factor
  that was used to generate the Image object from depth values.  If
  scale_factor is specified, it should be the same value that was
  specified in the original conversion.

  The result of this function should be equal to the original input
  within the precision of the conversion.

  Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

  Returns:
    A 2D floating point numpy array representing a depth image.

  """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 == len(image_shape)
    float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


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


def get_perception(step, suction_state, rgb, depth, lang_goal, camera_intri, camera_extri, clip_model):
    '''

    Args:
        step: start with 0, add 1 per frame
        rgb: H,W,C, int
        depth: H,W, int16
        lang_goal: str
        camera_intri: list [cx, cy, fx, fy]
        camera_extri: 7d list [x, y, z, ox, oy, oz, ow]
        clip_model
    Returns:

    '''
    obs = {}
    device = 'cuda:0'
    # front_camera
    cx, cy, fx, fy = camera_intri
    front_camera_intrinsics = np.array([[fx, 0., cx],
                                        [0., fy, cy],
                                        [0., 0., 1.],
                                        ])
    obs['front_camera_intrinsics'] = torch.tensor([front_camera_intrinsics], device=device).unsqueeze(0)

    front_camera_extrinsics = pose_to_4x4mat(camera_extri)
    obs['front_camera_extrinsics'] = torch.tensor([front_camera_extrinsics], device=device).unsqueeze(0)

    front_rgb = torch.tensor([copy.deepcopy(rgb)], device=device)
    front_rgb = front_rgb.permute(0, 3, 1, 2).unsqueeze(0)  # NHWC -> NCHW
    obs['front_rgb'] = front_rgb

    front_depth = copy.deepcopy(depth)
    front_depth = image_to_float_array(front_depth, 1000.0)  # mm -> m
    front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        front_depth,
        front_camera_extrinsics,
        front_camera_intrinsics)
    front_point_cloud = torch.tensor([front_point_cloud], device=device)
    np.save('vision_point_cloud.npy', front_point_cloud.detach().cpu().numpy())
    front_point_cloud = front_point_cloud.permute(0, 3, 1, 2).unsqueeze(0)  # NWHC -> NCHW
    obs['front_point_cloud'] = front_point_cloud

    tokens = clip.tokenize([lang_goal]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device)
    lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
    obs['lang_goal_embs'] = lang_embs.float()

    # proprio
    finger_positions = np.zeros(2)
    time = (1. - (step / float(10 - 1))) * 2. - 1.  # to do: modify this one
    low_dim_state = torch.tensor([[[suction_state,
                                    finger_positions[0],
                                    finger_positions[1],
                                    time]]])
    obs['low_dim_state'] = low_dim_state

    return obs, front_point_cloud.detach().cpu().numpy()


def init_peract(perceiver_actor_weight_path):
    # model init
    perceiver_actor = init_perceiver_actor(perceiver_actor_weight_path=perceiver_actor_weight_path)
    device = "cuda:0"
    clip_model, preprocess = clip.load("RN50", device=device)

    return perceiver_actor, clip_model


def inference_peract(step, suction_state, rgb, depth, preact, lang_goal, clip_model,
                     camera_intri=[64.4480685, 66.78173778, 182.29981142, 291.64949166],
                     camera_extri=np.array([
                         0.10431104317559005,
                         -0.45378789156378785,
                         0.6809180737974473,
                         0.9977884597380278,
                         -0.010699004055162784,
                         -0.06559005498610207,
                         0.0012905862086242488,
                     ])):
    # get observation then predict
    observation, pcd = get_perception(step,suction_state, rgb, depth, lang_goal=lang_goal, camera_intri=camera_intri,
                                      camera_extri=camera_extri,
                                      clip_model=clip_model)
    device = "cuda:0"
    observation = {k: v.to(device) for k, v in observation.items() if type(v) == torch.Tensor}
    prediction = preact.act(observation)
    return prediction, pcd


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


if __name__ == '__main__':
    # now the input are simulated
    perceiver_actor_weight_path = '/data1/peract/ckpts/gpt_uni_721_se3_minus_1/90000/'


    # rgb_image_path = '../valid/color/rgb_image_20230625_101016_769.jpg'
    # depth_image_path = "../valid/depth/depth_image_20230625_101016_769.png"

    rgb_image_path = "images/rgb.png"
    depth_image_path = "images/depth.png"

    step = 0  # += 1 per frame reset to 0 # very important
    suction_state = 1
    lang_goal = 'move poker'

    # rgb = np.array(Image.open(rgb_image_path))
    img_size = (128, 128)
    rgb = cv2.imread(rgb_image_path)[:, :, ::-1]  #
    if rgb.shape[:2] != img_size:
        rgb = cv2.resize(rgb, img_size, interpolation=cv2.INTER_NEAREST)
    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth.shape[:2] != img_size:
        depth = cv2.resize(depth, img_size, interpolation=cv2.INTER_NEAREST)

    if len(depth.shape) == 3:
        assert depth.shape[2] == 4
        depth = translate_8UC4_to_16UC1(depth)
        depth = depth.astype(np.float32)[:, :, 0]

    camera_intri = [64.4480685, 66.78173778, 182.29981142, 291.64949166]
    camera_extri = np.array([0.1087458851426165,
                                   -1.0974391229818714,
                                   0.5189316107518424,

                                   0.9396495772780166,
                                   0.023653515121718285,
                                   -0.055046279007725026,
                                   -0.336851733424418, ] )  # it has been rolled

    peract, clip_model = init_peract(perceiver_actor_weight_path)
    predication, pcd = inference_peract(step, suction_state, rgb, depth, peract, lang_goal, clip_model, camera_intri=camera_intri,
                                        camera_extri=camera_extri)

    # visualize

    rot = predication['pred_action']['rot_and_grip'][0][:3].detach().cpu()
    grip = predication['pred_action']['rot_and_grip'][0][3].detach().cpu()
    print('predicted gripper state: ', grip)
    quaternion = discrete_euler_to_quaternion(rot, 5)
    trans = predication['pred_action']['continuous_trans'][0][:3].detach().cpu()
    pose = np.concatenate((trans, quaternion))

    # you need a visualize window interface to see the pointclouds.
    #visualize_from_prediction(pcd[0,0].transpose(1,2,0).reshape(-1,3), pose)
    print(trans, rot, pose)
