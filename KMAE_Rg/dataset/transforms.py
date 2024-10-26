import os.path

import numpy as np
import medutils
import h5py
from torchvision.transforms import ColorJitter
from PIL import Image
import pandas as pd
from utils import crop_center2d, cartesian_mask_yt, generate_kdata_and_complex_image


class ToNpDtype(object):
    def __init__(self, key_val_pair):
        self.key_val_pair = key_val_pair

    def __call__(self, sample):
        for key, val in self.key_val_pair:
            sample[key] = sample[key].astype(val)
        return sample


class AddTorchChannelDim(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key][:, None]
        return sample

class AddBatchDim(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key][None]
        return sample


class MriOp(object):
    def __init__(self, mask, smaps):
        self.mask = mask
        self.smaps = smaps

    def forward(self, x):
        return medutils.mri.mriForwardOp(x, self.smaps, self.mask, soft_sense_dim=0)

    def adjoint(self, x):
        return medutils.mri.mriAdjointOp(x, self.smaps, self.mask, coil_axis=1)


class ComputeInit(object):
    def __call__(self, sample):
        op = MriOp(sample['masks'][None, ...], sample['smaps'][None, ...])
        optim = medutils.optimization.CgSenseReconstruction(op, max_iter=80)
        sample['init_new'], tol = optim.solve((sample['masks']*sample['kspace'])[None, ...], return_series=True, return_tol=True)

        return sample


class Normalize(object):
    def __init__(self, mode='2D', scale=1, axis=(-1, -2)):
        assert mode in ('2D', '3D')
        self.mode = mode
        self.scale = scale
        self.axis = axis

    def __call__(self, sample):
        if self.mode == '2D':
            min_2d = np.min(np.abs(sample['reference']), axis=self.axis)
            max_2d = np.max(np.abs(sample['reference']), axis=self.axis)
            if 'R' in sample:
                max_2d *= sample['R']
            for key in ['reference', 'kspace']:
                sample[key] = (sample[key] - min_2d[:, None, None])/((max_2d - min_2d)[:, None, None])
        elif self.mode == '3D':
            # max_3d = np.max(np.mean(np.abs(sample['init']), 0))  # todo: why use mean first?
            # min_3d = np.min(np.mean(np.abs(sample['init']), 0))
            max_3d = np.max(np.abs(sample['reference']))
            min_3d = np.min(np.abs(sample['reference']))
            if 'R' in sample:
                max_3d *= sample['R']
            for key in ['kspace', 'reference']:  # todo: normalize these three terms? # todo: state 16.02.22: no more normalization for kspace and reference now!!
                sample[key] = (sample[key])/(max_3d - min_3d) * self.scale
                # sample[key] = (sample[key] - min_3d)/(max_3d - min_3d) * self.scale
        return sample


class GeneratekSpace(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['kspace'], sample['reference'] = generate_kdata_and_complex_image(sample['reference'])
        return sample


class LoadMask(object):
    def __init__(self, pattern, R, data_root, dummy_batch_size=1, mode='train'):
        self.pattern = pattern  # if isinstance(pattern, list) else [pattern]
        self.R = R if isinstance(R, list) else [R]
        self.data_root = data_root
        self.dummy_batch_size = dummy_batch_size
        self.mode = mode

    def __call__(self, sample):
        R = np.random.choice(self.R)
        mask_idx_pool = np.random.choice(np.arange(20), size=self.dummy_batch_size, replace=False)
        masks = []
        nFE, nPE, nT = sample['reference'].shape
        for mask_idx in mask_idx_pool:
            pattern = self.pattern
            if R != 1:
                if self.mode == 'train':
                    # if pattern == 'VISTA':
                    #     mask_path = os.path.join(self.data_root, pattern, f"mask_VISTA_{nPE}x25_acc{R}_{mask_idx+1}.txt")
                    #     mask = np.loadtxt(mask_path, dtype=np.int32, delimiter=",")
                    if pattern == 'uniform' or 'normal-' in pattern:
                        sns = 10 if pattern == 'uniform' else int(pattern.split('-')[1])
                        mask = cartesian_mask_yt(nPE, nT, R, sample_n=4, mode=pattern, normal_sensitivity=sns).transpose()
                elif self.mode == 'val' or self.mode == 'infer':
                    mask_path = os.path.join('masks/', f"mask{nPE}x{nT}_R{np.random.choice(self.R)}_{self.pattern}.npy")
                    mask = np.load(mask_path)
            else:
                mask = np.ones((sample['nPE'], 25))  #, dtype=np.int32)
            masks.append(np.transpose(mask)[None, :, None, :])
        sample['masks'] = np.concatenate(masks, 0)
        sample['R'] = R
        return sample


class TemporalSubsampling(object):
    def __init__(self, subsampling_ratio, subsampling_list):
        self.subsampling_ratio = subsampling_ratio
        self.subsampling_list = subsampling_list

    def __call__(self, sample):
        for key in self.subsampling_list:
            sample[key] = sample[key][:, :, ::self.subsampling_ratio]
        return sample



class ExtractTimePatch(object):
    def __init__(self, patch_size, keys_dim_pair, mode='val'):
        self.patch_size = patch_size
        self.keys_dim_pair = keys_dim_pair
        self.mode = mode

    def __call__(self, sample):
        frames = sample[self.keys_dim_pair[0][0]].shape[self.keys_dim_pair[0][1]]
        if self.mode == 'train':
            start_idx = np.random.randint(0, frames-self.patch_size if frames-self.patch_size > 0 else 1)
        else:
            start_idx = 0

        for key, dim in self.keys_dim_pair:
            sample[key] = np.swapaxes(sample[key], 0, dim)
            sample[key] = sample[key][start_idx:start_idx+self.patch_size]
            sample[key] = np.swapaxes(sample[key], 0, dim)

        return sample


class SwapDim(object):
    def __init__(self, keys, swap_axis):
        assert len(keys) == len(swap_axis)
        self.keys = keys
        self.swap_axis = swap_axis

    def __call__(self, sample):
        for key, swap_axes in zip(self.keys, self.swap_axis):
            sample[key] = np.swapaxes(sample[key], *swap_axes)
        return sample


class Transpose(object):
    def __init__(self, keys, transpose_axis):
        assert len(keys) == len(transpose_axis)
        self.keys = keys
        self.transpose_axis = transpose_axis

    def __call__(self, sample):
        for key, transpose_axes in zip(self.keys, self.transpose_axis):
            sample[key] = np.transpose(sample[key], transpose_axes)
        return sample
