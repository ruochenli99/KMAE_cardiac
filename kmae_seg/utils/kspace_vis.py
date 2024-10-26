import os
import numpy as np
import pandas as pd
import h5py
from utils import multicoil2single, image_normalization
import matplotlib.pyplot as plt
import torch

data_root = '/home/data/CINE_134_h5/'
csv_file = '../dataset/csv_files/CINE2D_192x156.csv'
csv_file = '/home/peter/PycharmProjects/moco_reconstruction_cine2d/dataset/csv_files/134Samples/CINE2D_All.csv'
selected_slices = 'No_occ_slice'
save_dir = '/home/data/CINE_134_h5_vis/kspace_single'
save_dir = '/home/data/CINE_134_h5_vis/all_kspace_single'

subjs_csv = pd.read_csv(csv_file)

data_names = [fname for fname in subjs_csv.filename]
data_names = [fname.split('.')[0] for fname in data_names]
data_paths = [os.path.join(data_root, f'{name}.h5') for name in data_names]

valid_slice = [np.arange(s_start, s_end) for s_start, s_end in zip(eval(f'subjs_csv.{selected_slices}_start'), eval(f'subjs_csv.{selected_slices}_end'))]

for i, name in enumerate(data_names):
    slc = valid_slice[i][len(valid_slice[i])//2]
    path = data_paths[i]
    with h5py.File(path, 'r', swmr=True, libver='latest') as ds:
        kspace = ds['kSpace'][slc].astype(np.complex64).transpose(0, 1, 3, 2)
        smaps = ds['dMap'][slc].astype(np.complex64).transpose(0, 1, 3, 2)
    kspace = torch.tensor(kspace)
    smaps = torch.tensor(smaps)
    kspace_single = multicoil2single(kspace, smaps)
    vis = image_normalization(torch.log(kspace_single.abs()[None] + 0.0001).cpu().numpy().transpose(1, 0, 2, 3), 255,
                        mode='3D').astype(np.uint8)
    plt.figure()
    plt.imshow(vis[0, 0])
    plt.savefig(os.path.join(save_dir, f'{name}_k_{slc}.png'))
    plt.close()








