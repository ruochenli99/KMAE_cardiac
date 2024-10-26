import numpy as np
import medutils
from pathlib import Path
import os
import pandas as pd


def save_individual(imgs, save_dir, normalize=True):
    for i, img in enumerate(imgs):
        save_path = os.path.join(save_dir, f'f{i}.png')
        medutils.visualization.imsave(img, save_path, normalize_img=normalize)


def plot_array(img, M=None, N=None):
    """
    flatten the images in the size either of B x M X N (single channel) or B x M X N x 3 (RGB channel).
    This function is modified based on the function from plot_array in medutils
    :param img:
    :param M:
    :param N:
    :return:
    """
    assert img.ndim in (3, 4)
    if img.ndim == 4:
        assert img.shape[-1] == 3

    ksz_M = img.shape[1]
    ksz_N = img.shape[2]

    if M is None or N is None:
        M = int(np.floor(np.sqrt(img.shape[0])))
        N = int(np.ceil(img.shape[0] / M))
    else:
        assert M * N == img.shape[0]

    arr = np.zeros((M * ksz_M, N * ksz_N), dtype=img.dtype) if img.ndim == 3 else np.zeros((M * ksz_M, N * ksz_N, 3), dtype=img.dtype)
    for i in range(img.shape[0]):
        ii = np.mod(i, M)
        jj = i // M
        arr[ii * ksz_M:ii * ksz_M + ksz_M, jj * ksz_N:jj * ksz_N + ksz_N] = img[i]
    return arr


def cal_metric_avg(metric_paths_list, save_path, fmt=None, header=None):
    all_data = []
    column_names = list(pd.read_csv(metric_paths_list[0]).columns)
    for metric_path in metric_paths_list:
        metric_data = pd.read_csv(metric_path)
        assert list(metric_data.columns) == column_names, 'the columns names do not match to each other'
        metric_data = metric_data.to_numpy()

        all_data.append(metric_data)
    all_data = np.concatenate(all_data)
    if column_names[0] == '# frame':
        # column_names = column_names[1:]
        all_data = all_data[:, 1:]
    nframe = len(all_data)
    avg = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    values = np.hstack((nframe, avg, std))
    if not fmt:
        fmt = ['%d', '%1.4f', '%1.2f', '%1.4f', '%1.4f', '%1.4f', '%1.4f']
    if not header:
        header = 'frame,SSIM,PSNR,NRMSE,SSIM_std,PSNR_std,NRMSE_std'
    np.savetxt(save_path, values[None, ...], fmt=fmt, delimiter=',', header=header)



if __name__ == '__main__':
    metric_paths = [
        '/home/peter/results/debug_0203/Pat11/trR12_teR12/R12_best/metrics_all/moco_s8_i1.csv',
        '/home/peter/results/debug_0203/Pat11/trR12_teR12/R12_best/metrics_all/moco_s8_i2.csv',
    ]
    save_path = '/home/peter/results/debug_0203/Pat11/trR12_teR12/R12_best/test.csv'
    cal_metric_avg(metric_paths, save_path)
