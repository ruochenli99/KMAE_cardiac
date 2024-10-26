import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Union, Optional, Tuple, Callable


def generate_kdata_and_complex_image(frame_im_data: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Add synthetic phase information into image data and simulate k-space measurements.
    """
    img = frame_im_data/np.max(frame_im_data)  # Normalize image intensity to range [0, 1]

    img_ = np.transpose(img)
    img_complex_ = SimulateCartesian()(img_)
    k_space_ = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_complex_, axes=(-2, -1)), norm='ortho', axes=(-2, -1)), axes=(-2, -1))
    k_space = np.transpose(k_space_)
    # k_space = normalize_kspace(k_space) # TODO decide whether to normalize kspace data
    img_complex = np.transpose(img_complex_)
    return k_space, img_complex

class SimulateCartesian(object):

    @staticmethod
    def _fspecial(shape, sigma):
        """
        2D gaussian masks - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    @staticmethod
    def _transform_kspace_to_image(k, dim=None, img_shape=None):
        """ Computes the Fourier transform from k-space to image space
        along a given or all dimensions
        :param k: k-space data
        :param dim: vector of dimensions to transform
        :param img_shape: desired shape of output image
        :returns: data in image space (along transformed dimensions)
        """
        if not dim:
            dim = range(k.ndim)

        img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
        img *= np.sqrt(np.prod(np.take(img.shape, dim)))
        return img

    def _addSynthethicPhase(self, img):
        """
        Add synthetic phase to real-valued image. The phase is sampled from gaussian B0 variations (similar to LORAKS).
        """
        b0 = np.random.normal(4, 0.5, (img.shape[-2], img.shape[-1]))
        smoother = self._fspecial((img.shape[-2], img.shape[-1]), max(1.0, round(
            img.shape[-2] / 200)))  # TODO: Ask Kerstin what this 200 is for, had to add min(1.0,...)
        b00 = self._transform_kspace_to_image(b0 * smoother)
        img = img * np.exp(1j * np.angle(b00))
        return img

    def __call__(self, arg):
        img = self._addSynthethicPhase(arg).astype(np.complex64)
        img = np.ascontiguousarray(img)
        return img


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask_yt(ky, dim_t, acc, sample_n=5, centred=True, mode='uniform', normal_sensitivity=10):
    mask_stack = []
    for t in range(dim_t):
        mask = cartesian_mask((1, ky, 1), acc, sample_n=sample_n, centred=centred, mode=mode, normal_sensitivity=normal_sensitivity)
        mask_stack.append(mask)
    mask_stack = np.concatenate(mask_stack, axis=0)
    return mask_stack.squeeze()

def cartesian_mask_yt_uniform(ky, dim_t, acc):
    mask_stack = []
    for t in range(dim_t):
        idx = np.random.choice(ky, int(ky/acc), False)
        mask = np.zeros((1, ky, 1))
        mask[0, idx, 0] = 1
        mask_stack.append(mask)
    mask_stack = np.concatenate(mask_stack, axis=0)
    return mask_stack.squeeze()

def cartesian_mask(shape, acc, sample_n=10, centred=True, mode='uniform', normal_sensitivity=10):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/normal_sensitivity)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n
    if mode == 'uniform':
        pdf_x = np.ones(Nx) / (Nx - sample_n)
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0


    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, p=pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask


def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm='ortho')


def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm='ortho')


def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)


def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)


class MulticoilForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = torch.unsqueeze(image[:,0], self.coil_axis) * smaps
        else:
            coilimg = torch.unsqueeze(image, self.coil_axis) * smaps
        kspace = self.fft2(coilimg)
        masked_kspace = kspace * mask
        return masked_kspace


class MulticoilAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, kspace, mask, smaps):
        masked_kspace = kspace * mask
        coilimg = self.ifft2(masked_kspace)
        img = torch.sum(torch.conj(smaps) * coilimg, self.coil_axis)

        if self.channel_dim_defined:
            return torch.unsqueeze(img, 1)
        else:
            return img


class ForwardOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2

    def forward(self, image, mask):
        kspace = self.fft2(image)
        masked_kspace = kspace * mask
        return masked_kspace


class AdjointOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2

    def forward(self, kspace, mask):
        masked_kspace = kspace * mask
        img = self.ifft2(masked_kspace)
        return img



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mask = cartesian_mask_yt(156, 16, 4, sample_n=4, centred=True, uniform=True)

    # mask_stack = []
    # for i in range(16):
    #     masks = cartesian_mask((1, 156, 1), 4, sample_n=6, centred=True, uniform=True)
    #     mask_stack.append(masks)
    # mask_stack = np.concatenate(mask_stack, axis=0)
    # plt.imshow(mask_stack[..., 0])
    plt.imshow(mask)
    plt.show()

