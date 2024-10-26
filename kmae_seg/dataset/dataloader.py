import torch
import pandas as pd
import numpy as np
import torchvision
from utils import ToTorchIO
from dataset.transforms import *

import time
from typing import Union, Optional
from pathlib import Path
import nibabel as nib


class UKBioBankKMAE(torch.utils.data.Dataset):
    def __init__(self, config, mode, subject_list):
        self.data_root = config.data_root
        self.mode = mode
        self.subj_list = load_img_path(load_dir=self.data_root, slices=config.slices, subject_list=subject_list)
        self.cropped_img = config.cropped_img

        self.transform = torchvision.transforms.Compose(self.get_transform(config=config))

    def __len__(self):
        return len(self.subj_list)

    def _apply_transform(self, sample):
        return self.transform(sample)

    def assign_seg_label(self, seg, current_label=1.0):
        loc = seg == current_label
        seg = np.zeros_like(seg)
        seg[loc] = 1
        return seg

    def __getitem__(self, idx):
        img_path = self.subj_list[idx][0]
        slc = self.subj_list[idx][1]
        seg_path = img_path.parent / "seg_sa_cropped.nii.gz"
        if self.cropped_img:
            img_path = img_path.parent / "sa_cropped.nii.gz"
            seg_path = img_path.parent / "seg_sa_cropped.nii.gz"

        subj_id = img_path.parent.name
        img = nib.load(img_path).get_fdata()[..., slc, :]
        seg = nib.load(seg_path).get_fdata()[..., slc, :]
        assert img.shape == seg.shape, f"Image and segmentation shapes do not match for subj {subj_id}: {img.shape} vs {seg.shape}"

        current_label = 2.0
        seg = self.assign_seg_label(seg, current_label)

        # header = nib.load(img_path).header
        img /= np.max(img)

        d = {"reference": img, "seg": seg}
        sample = self._apply_transform(d)

        # augmentation # todo
        return sample, subj_id, slc

    def get_transform(self, config):
        assert self.mode in ('train', 'val', 'infer')

        data_transforms = []

        data_transforms.append(TemporalSubsampling(config.temporal_subsample, ['reference', 'seg']))
        data_transforms.append(LoadMask(config.mask_pattern, config.acc_rate, config.mask_root, mode=self.mode))
        data_transforms.append(GeneratekSpace())
        # data_transforms.append(ToNpDtype([('reference', np.complex64), ('kspace', np.complex64),
        #                                   ('seg', np.float32), ('masks', np.float32), ]))
        data_transforms.append(ToNpDtype([('reference', np.float32), ('kspace', np.complex64),
                                          ('seg', np.float32), ('masks', np.float32), ]))
        data_transforms.append(Transpose(('reference', 'kspace', 'seg'), [(2,1,0), (2,1,0), (2,1,0)]))
        input_convert_list = ['kspace', 'masks', 'seg']
        data_transforms.append(ToTorchIO(input_convert_list, ['reference']))

        return data_transforms


def load_img_path(load_dir: Union[str, Path],
                    subject_list: Optional[list],
                    slices='center', # 'center', 'all', center3, center5
                    img_file_name="sa_cropped.nii.gz",
                    seg_file_name="seg_sa_cropped.nii.gz",
                    ):
    data_list = []
    count = 0
    start_time = time.time()

    for i, subject in enumerate(subject_list):
        im_path = Path(os.path.join(load_dir, subject, img_file_name))
        seg_path = Path(os.path.join(load_dir, subject, seg_file_name))

        # Make sure both image and segmentation files exist
        if not os.path.exists(im_path) or not os.path.exists(seg_path):
            continue
        try:
            im = nib.load(im_path)
            if im.shape[2] < 9 or im.shape[3] != 50:
                continue
            if slices == 'center':
                idx = [im.shape[2] // 2]
            elif slices == 'all':
                idx = list(range(im.shape[2]))
            elif slices == 'center3':
                idx = [im.shape[2] // 2 - 1, im.shape[2] // 2, im.shape[2] // 2 + 1]
            elif slices == 'center5':
                idx = [im.shape[2] // 2 - 2, im.shape[2] // 2 - 1, im.shape[2] // 2, im.shape[2] // 2 + 1, im.shape[2] // 2 + 2]

            for i in idx:
                data_list.append([im_path, i, im.header.get_zooms()])
        except IOError:
            print(f"Failed to load image/segmentation pair: {im_path}")
            continue

        count += 1

    elapsed = time.time() - start_time
    print(f"Found {count} cases in {elapsed}s.")
    return data_list



