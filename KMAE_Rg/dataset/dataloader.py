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
        # self.subj_list = self.subj_list[:2] +self.subj_list[-2:]
        self.cropped_img = config.cropped_img
        self.rg_class=config.rg_class


        if self.rg_class=='age':
            self.df = pd.read_csv('cardia_age.csv')
            self.df['eid'] = self.df['eid'].astype(str)
            self.df_dict = self.df.set_index('eid')['age'].to_dict()
        if self.rg_class=='LVEDV':
            self.df = pd.read_csv('age_LVEDV_LVEF.csv')
            self.df['eid'] = self.df['eid'].astype(str)
            self.df_dict = self.df.set_index('eid')['LVEDV'].to_dict()
        if self.rg_class=='LVEF':
            self.df = pd.read_csv('age_LVEDV_LVEF.csv')
            self.df['eid'] = self.df['eid'].astype(str)
            self.df_dict = self.df.set_index('eid')['LVEF'].to_dict()
        # self.transform = torchvision.transforms.Compose(self.get_transform_image(config=config))
        self.transform = torchvision.transforms.Compose(self.get_transform(config=config))


    def __len__(self):
        # print(len(self.subj_list))
        return len(self.subj_list)

    def _apply_transform(self, sample):
        return self.transform(sample)

    def __getitem__(self, idx):
        sample_list = []
        for i in range(len(self.subj_list[idx])):
            img_path = self.subj_list[idx][i][0]
            slc = self.subj_list[idx][i][1]
            seg_path = img_path.parent / "sa_cropped.nii.gz"
            if self.cropped_img:
                img_path = img_path.parent / "sa_cropped.nii.gz"
                seg_path = img_path.parent / "sa_cropped.nii.gz"

            subj_id = img_path.parent.name
            img = nib.load(img_path).get_fdata()[..., slc, :]
            seg = nib.load(seg_path).get_fdata()[..., slc, :]
            assert img.shape == seg.shape, f"Image and segmentation shapes do not match for subj {subj_id}: {img.shape} vs {seg.shape}"
            # header = nib.load(img_path).header
            img /= np.max(img)
            # label = self.df_dict[subj_id]
            label = torch.tensor(float(self.df_dict[subj_id]))

            d = {"reference": img, "seg": seg}
            sample = self._apply_transform(d)
            sample_list.append(sample)

        new_sample = [[[], [], []], [[]]]
        for sample in sample_list:
            new_sample[0][0].append(sample[0][0])
            new_sample[0][1].append(sample[0][1])
            new_sample[0][2].append(sample[0][2])
            new_sample[1][0].append(sample[1][0])


        new_sample[0][0] = torch.stack(new_sample[0][0])
        new_sample[0][1] = torch.stack(new_sample[0][1])
        new_sample[0][2] = torch.stack(new_sample[0][2])
        new_sample[1][0] = torch.stack(new_sample[1][0])

        # augmentation # todo
        return new_sample, subj_id, slc,label,img

    def get_transform(self, config):
        assert self.mode in ('train', 'val', 'infer')

        data_transforms = []

        data_transforms.append(TemporalSubsampling(config.temporal_subsample, ['reference', 'seg']))
        data_transforms.append(LoadMask(config.mask_pattern, config.acc_rate, config.mask_root, mode=self.mode))
        data_transforms.append(GeneratekSpace())
        data_transforms.append(ToNpDtype([('reference', np.complex64), ('kspace', np.complex64),
                                          ('seg', np.float32), ('masks', np.float32), ]))
        data_transforms.append(Transpose(('reference', 'kspace', 'seg'), [(2,1,0), (2,1,0), (2,1,0)]))

        input_convert_list = ['kspace', 'masks', 'seg']
        data_transforms.append(ToTorchIO(input_convert_list, ['reference']))

        return data_transforms

    def get_transform_image(self, config):
        assert self.mode in ('train', 'val', 'infer')

        data_transforms = []

        data_transforms.append(TemporalSubsampling(config.temporal_subsample, ['reference', 'seg']))
        data_transforms.append(LoadMask(config.mask_pattern, config.acc_rate, config.mask_root, mode=self.mode))
        # data_transforms.append(GeneratekSpace())
        data_transforms.append(ToNpDtype([('reference', np.float32),
                                          ('seg', np.float32), ('masks', np.float32), ]))
        data_transforms.append(Transpose(('reference',  'seg'), [(2,1,0),  (2,1,0)]))

        input_convert_list = ['reference', 'masks', 'seg']
        data_transforms.append(ToTorchIO(input_convert_list, ['reference']))

        return data_transforms


def load_img_path(load_dir: Union[str, Path],
                    subject_list: Optional[list],
                    slices='center', # 'center', 'all', center3, center5
                    img_file_name="sa_cropped.nii.gz",
                    seg_file_name="sa_cropped.nii.gz",
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

            subject_data_list = []
            for i in idx:
                subject_data_list.append([im_path, i, im.header.get_zooms()])
            data_list.append(subject_data_list)
        except IOError:
            print(f"Failed to load image/segmentation pair: {im_path}")
            continue

        count += 1

    elapsed = time.time() - start_time
    print(f"Found {count} cases in {elapsed}s.")
    return data_list



