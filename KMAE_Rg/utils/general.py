import numpy as np
import torch
import json
import medutils
import os
import matplotlib.pyplot as plt
import lpips
import os
import json


def fix_dict_in_wandb_config(wandb):
    """"Adapted from [https://github.com/wandb/client/issues/982]"""
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            keys = k.split('.')
            if len(keys) == 2:
                new_key = k.split('.')[0]
                inner_key = k.split('.')[1]
                if new_key not in config.keys():
                    config[new_key] = {}
                config[new_key].update({inner_key: v})
                del config[k]
            elif len(keys) == 3:
                new_key_1 = k.split('.')[0]
                new_key_2 = k.split('.')[1]
                inner_key = k.split('.')[2]

                if new_key_1 not in config.keys():
                    config[new_key_1] = {}
                if new_key_2 not in config[new_key_1].keys():
                    config[new_key_1][new_key_2] = {}
                config[new_key_1][new_key_2].update({inner_key: v})
                del config[k]
            else: # len(keys) > 3
                raise ValueError('Nested dicts with depth>3 are currently not supported!')

    wandb.config = wandb.Config()
    for k, v in config.items():
        wandb.config[k] = v


def normalize_np(img, vmin=None, vmax=None, max_int=255.0):
    """ normalize (magnitude) image
    :param image: input image (np.array)
    :param vmin: minimum input intensity value
    :param vmax: maximum input intensity value
    :param max_int: maximum output intensity value
    :return: normalized image
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())
    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)
    img = (img - vmin)*(max_int)/(vmax - vmin)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img


def imsave_custom(img, filepath, normalize_img=True, vmax=None):
    """ Save (magnitude) image in grayscale
    :param img: input image (np.array)
    :param filepath: path to file where k-space should be save
    :normalize_img: boolean if image should be normalized between [0, 255] before saving
    """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize_img:
        img = normalize_np(img, vmax)
    plt.imsave(filepath, img)


def get_metric(name):
    if name == 'PSNR':
        return medutils.measures.psnr
    elif name == 'SSIM':
        return medutils.measures.ssim
    elif name == 'NRMSE':
        return medutils.measures.nrmseAbs
    elif name == 'LPIPS':
        return lpips.LPIPS(net='alex')
    else:
        raise NotImplementedError


def pad(inp, divisor=8):
    pad_x = int(np.ceil(inp.shape[-2] / divisor)) * divisor - inp.shape[-2]
    pad_y = int(np.ceil(inp.shape[-1] / divisor)) * divisor - inp.shape[-1]
    inp = torch.nn.functional.pad(inp, (pad_y, 0, pad_x, 0))
    return inp, {'pad_x': pad_x, 'pad_y': pad_y}


def unpad(inp, pad_x, pad_y):
    return inp[..., pad_x:, pad_y:]


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def image_normalization(image, scale=1, mode='2D'):
    if mode == '2D':
        return scale * (image - np.min(image)) / (np.max(image) - np.min(image))
    elif mode == '3D':
        if np.iscomplexobj(image):
            image = np.abs(image)
        max_3d = np.max(image)
        min_3d = np.min(image)
        return scale * (image - min_3d) / (max_3d - min_3d)
    else:
        raise NotImplementedError


def image_normalization_torch(image, scale=1, mode='3D'):
    if mode == '3D':
        max_3d = image.abs().max()
        min_3d = image.abs().min()
        image = (image - min_3d) / (max_3d - min_3d) * scale
    else:
        raise NotImplementedError
    return image


def crop_center(imgs, cropx, cropy):
    # todo: need to be refactor!! currently badly written
    if not isinstance(imgs, list):
        imgs = [imgs]
    out = []
    for img in imgs:
        if len(img.shape) == 2:
            x, y = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            out.append(img[startx:startx + cropx, starty:starty + cropy])
        elif len(img.shape) == 4:
            _, _, x, y = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            out.append(img[..., startx:startx + cropx, starty:starty + cropy])
        elif len(img.shape) == 3:
            x, y, _ = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            out.append(img[startx:startx + cropx, starty:starty + cropy, ...])
    if len(out) == 1:
        out = out[0]
    return out


def crop_center2d(imgs, crop_size_x_y, crop_dim_x_y):
    cropx, cropy = crop_size_x_y[0], crop_size_x_y[1]
    crop_dim_x, crop_dim_y = crop_dim_x_y[0], crop_dim_x_y[1]
    assert imgs.shape[crop_dim_x] >= cropx and imgs.shape[crop_dim_y] >= cropy
    shape_x, shape_y = imgs.shape[crop_dim_x], imgs.shape[crop_dim_y]
    startx = shape_x // 2 - (cropx // 2)
    starty = shape_y // 2 - (cropy // 2)

    # todo: how to replace this block? is there a certain function to choose along a certain axis?
    imgs = np.swapaxes(imgs, 0, crop_dim_x)
    imgs = imgs[startx:startx + cropx, ...]
    imgs = np.swapaxes(imgs, 0, crop_dim_x)
    imgs = np.swapaxes(imgs, 0, crop_dim_y)
    imgs = imgs[starty:starty + cropy, ...]
    imgs = np.swapaxes(imgs, 0, crop_dim_y)

    return imgs


def crop_center2d_torch(imgs, crop_size_x_y):
    cropx, cropy = crop_size_x_y[0], crop_size_x_y[1]
    assert imgs.shape[-2] >= cropx and imgs.shape[-1] >= cropy
    shape_x, shape_y = imgs.shape[-2], imgs.shape[-1]
    startx = shape_x // 2 - (cropx // 2)
    starty = shape_y // 2 - (cropy // 2)

    imgs = imgs[..., startx:startx + cropx, starty:starty + cropy]
    return imgs


def warp_torch(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    flo = torch.flip(flo, dims=[1])
    # vgrid = Variable(grid) + flo
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    # masks = torch.autograd.Variable(torch.ones(x.size())).cuda()

    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # if W==128:
    # np.save('masks.npy', masks.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def generate_color_encoding(add_circle=True):
    x = np.linspace(-1, 1, 101)
    y = np.linspace(-1, 1, 101)
    xv, yv = np.meshgrid(x, y)
    coord = np.concatenate((xv[..., None], yv[..., None]), axis=-1)

    import flow_vis
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    color_encoding = flow_vis.flow_to_color(coord)
    if add_circle:
        circle1 = plt.Circle((50, 50), 50, color='white', fill=False)
        ax.add_patch(circle1)
    ax.imshow(color_encoding)
    ax.axis('off')
    plt.show()


def neighboring_frame_select(input, slc, neighboring_frame, frame_dim=1):
    """
    the input is regarded as cyclic.
    :param input:
    :param slc:
    :param neighboring_frame:
    :param frame_dim:
    :return:
    """
    nfr = input.shape[frame_dim]
    if isinstance(neighboring_frame, int): assert 2*neighboring_frame+1 <= nfr
    # alternative: for neighboring_frame == 'all' we can also shift nothing
    shift_offset = int(nfr/2) - slc if neighboring_frame == 'all' else neighboring_frame - slc

    input_shifted = torch.roll(input, shift_offset, dims=frame_dim)
    output = torch.swapaxes(input_shifted, frame_dim, 0)
    # del input_shifted
    # torch.cuda.empty_cache()
    # output = torch.swapaxes(input.roll(shift_offset, dims=frame_dim), frame_dim, 0)
    if isinstance(neighboring_frame, int):
        output = output[:2*neighboring_frame+1, ...]
    output = torch.swapaxes(output, 0, frame_dim)
    return output


# haven't been applied in the code
def neighboring_frame_for_all_slice(data, neighbor_num, frame_dim, new_dim, slices=None):
    data_list = []
    if slices is None:
        total_f = data.shape[frame_dim]
        slices = range(total_f)
    for slc in slices:
        neighbor_data = neighboring_frame_select(data, slc, neighbor_num, frame_dim=frame_dim)
        neighbor_data = neighbor_data.unsqueeze(new_dim)
        data_list.append(neighbor_data)
    return torch.cat(data_list, dim=new_dim)


def cal_lstsq_error(ref, recon):
    ref_flat = ref.flatten()
    recon_flat = recon.flatten()

    # calculate the least-squares solution
    recon2ch = np.concatenate((recon_flat.real, recon_flat.imag))
    ref2ch = np.concatenate((ref_flat.real, ref_flat.imag))
    slope, resid = np.linalg.lstsq(np.stack([recon2ch, np.ones_like(recon2ch)], axis=1), ref2ch, rcond=None)[0]

    recon = recon * slope + resid

    error = np.abs(np.abs(recon) - np.abs(ref))
    return error
