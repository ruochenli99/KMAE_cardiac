import os
import numpy as np
import medutils
from utils import plot_array, save_individual, image_normalization, cal_lstsq_error, error_map_preprocess_wandb
from pathlib import Path
import matplotlib.pyplot as plt
import torch
# from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast
import wandb
from medutils.measures import ssim, psnr, nrmse
from losses import LPIPSLoss


class InferenceWandb:
    def __init__(self, save_table_column=None, save_item=None, save_local=False, save_local_item=None, save_dir=None):
        self.save_table_column = save_table_column
        self.save_local = save_local
        self.save_local_item = save_local_item
        self.save_item = save_item
        self.data_list = []
        self.save_dir = save_dir


class InferenceLog(InferenceWandb):
    def __init__(self, save_table_column=None, save_img=False, save_local=False, save_dir=None, save_local_item=None, im_crop_size=None, save_mean_std=None):
        super().__init__(save_table_column, save_local=save_local, save_local_item=save_local_item, save_dir=save_dir)
        self.im_crop_size = im_crop_size
        self.save_mean_std = save_mean_std
        if 'LPIPS' in save_table_column:
            self.lpips = LPIPSLoss().cuda()
        if not save_img:
            remove_list = ['ref', 'recon', 'recon_baseline1', 'error_map', 'ref_yt', 'recon_yt', 'recon_baseline1_yt', 'error_map_yt']
            for item in remove_list:
                try:
                    self.save_table_column.remove(item)
                except:
                    pass

    def log_table(self):
        test_table = wandb.Table(data=self.data_list, columns=self.save_table_column)
        wandb.log({'test_table': test_table}, commit=True)

    def log_mean_std(self):
        cols = [self.save_table_column.index(i) for i in self.save_mean_std]
        mean = np.mean(np.array([[s[col] for col in cols] for s in self.data_list]), axis=0)
        std = np.std(np.array([[s[col] for col in cols] for s in self.data_list]), axis=0)
        mean_std = np.concatenate((mean[None], std[None]), axis=0)

        wandb.log({"Mean&Std": wandb.Table(data=mean_std, columns=self.save_mean_std)}, commit=False)

    def save(self, names, refs, recons, mask_boundary=None, motions=None, **kwargs):
        if self.save_table_column is not None:
            yt_flag_recon = 1
            yt_flag_ref = 1
            yt_flag_error_map = 1
            if mask_boundary is not None:
                xmin, xmax, ymin, ymax = mask_boundary
            else:
                xmin, xmax, ymin, ymax = 0, refs.shape[2], 0, refs.shape[3]
            xmean, ymean = int((xmin+xmax)/2), int((ymin+ymax)/2)
            if motions is not None:
                motion = self.torch2np(motions).transpose(0,2,3,1)
            if self.im_crop_size is not None:
                vxmin, vxmax, vymin, vymax = self.im_crop_size
            else:
                vxmin, vxmax, vymin, vymax = 0, refs.shape[2], 0, refs.shape[3]

            error = cal_lstsq_error(refs.cpu().numpy().transpose(1, 2, 3, 0), recons.cpu().numpy().transpose(1, 2, 3, 0))
            error_w = error_map_preprocess_wandb(error, eval_error_map_vmax=kwargs['eval_error_map_vmax'])
            refs_vis = image_normalization(refs.abs().cpu().numpy().transpose(1, 2, 3, 0), 255, mode='3D').astype(np.uint8)
            refs_vis_yt = refs_vis[:, :, ymean, :].transpose(1, 0, 2)
            recons_vis = image_normalization(recons.abs().cpu().numpy().transpose(1, 2, 3, 0), 255, mode='3D').astype(np.uint8)
            recons_vis_yt = recons_vis[:, :, ymean, :].transpose(1, 0, 2)
            error_vis_yt = error[:, :, ymean, :].transpose(1, 0, 2)
            error_vis_yt_w = error_w[:, :, ymean, :].transpose(1, 0, 2)

            for i in range(refs.shape[1]):
                my_data = []
                name = names+f'/{i}'
                my_data.append(name)
                if self.save_local:
                    save_dir_new = self.save_dir + '_' + names.split('/')[1]
                    name_save = '_'.join(name.split('/')[2:])
                    Path(save_dir_new).mkdir(parents=True, exist_ok=True)
                for key in self.save_table_column:
                    if key == 'ref':
                        my_data.append(wandb.Image(refs_vis[i], caption=f'ref_{name}'))
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/ref_{name_save}.png', refs_vis[i, vxmin:vxmax, vymin:vymax].squeeze(), cmap='gray')
                    elif key == 'recon':
                        my_data.append(wandb.Image(recons_vis[i], caption=f'recon_{name}'))
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/recon_{name_save}.png', recons_vis[i, vxmin:vxmax, vymin:vymax].squeeze(), cmap='gray')
                    elif key == 'error_map':
                        my_data.append(wandb.Image(error_w[i], caption=f'error_map_{name}'))
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/error_map_{name_save}.png', error[i,vxmin:vxmax, vymin:vymax].squeeze(),
                                       vmax=kwargs['eval_error_map_vmax'])
                    elif key == 'PSNR':
                        my_data.append(psnr(self.torch2np(recons[0, i]), self.torch2np(refs[0, i])))
                    elif key == 'SSIM':
                        my_data.append(ssim(self.torch2np(recons[0, i]), self.torch2np(refs[0, i])))
                    elif key == 'NRMSE':
                        my_data.append(nrmse(self.torch2np(recons[0, i]), self.torch2np(refs[0, i])))
                    elif key == 'LPIPS':
                        my_data.append(self.lpips(recons[:, i:i+1], refs[:, i:i+1]).item())
                    elif key == 'PSNR_center':
                        my_data.append(psnr(self.torch2np(recons[0, i, xmin:xmax, ymin:ymax]), self.torch2np(refs[0, i, xmin:xmax, ymin:ymax])))
                    elif key == 'SSIM_center':
                        my_data.append(ssim(self.torch2np(recons[0, i, xmin:xmax, ymin:ymax]), self.torch2np(refs[0, i, xmin:xmax, ymin:ymax])))
                    elif key == 'NRMSE_center':
                        my_data.append(nrmse(self.torch2np(recons[0, i, xmin:xmax, ymin:ymax]), self.torch2np(refs[0, i, xmin:xmax, ymin:ymax])))
                    elif key == 'LPIPS_center':
                        my_data.append(self.lpips(recons[:, i:i+1, xmin:xmax, ymin:ymax], refs[:, i:i+1, xmin:xmax, ymin:ymax]).item())
                    elif key == 'PSNR_yt':
                        my_data.append(psnr(self.torch2np(recons[0, :, :, ymean]), self.torch2np(refs[0, :, :, ymean])))
                    elif key == 'ref_yt':
                        my_data.append(wandb.Image(refs_vis_yt, caption=f'ref_yt_{name}')) if yt_flag_ref else my_data.append(wandb.Image(np.ones((2, 2, 1))))
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/ref_yt.png', refs_vis_yt[vxmin:vxmax].squeeze(), cmap='gray')
                        yt_flag_ref = 0
                    elif key == 'recon_yt':
                        my_data.append(wandb.Image(recons_vis_yt, caption=f'recon_yt_{name}')) if yt_flag_recon else my_data.append(wandb.Image(np.ones((2, 2, 1))))
                        # my_data.append(wandb.Image(recons_vis_yt))
                        yt_flag_recon = 0
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/recon_yt.png', recons_vis_yt[vxmin:vxmax].squeeze(), cmap='gray')
                    elif key == 'error_map_yt':
                        my_data.append(wandb.Image(error_vis_yt_w, caption=f'error_map_yt_{name}')) if yt_flag_error_map else my_data.append(wandb.Image(np.ones((2, 2, 1))))
                        yt_flag_error_map = 0
                        if self.save_local and key in self.save_local_item:
                            plt.imsave(f'{save_dir_new}/error_map_yt.png', error_vis_yt[vxmin:vxmax].squeeze(),
                                       vmax=kwargs['eval_error_map_vmax'])
                    else:
                        continue
                self.data_list.append(my_data)

    @staticmethod
    def torch2np(tensor):
        return tensor.detach().cpu().numpy()