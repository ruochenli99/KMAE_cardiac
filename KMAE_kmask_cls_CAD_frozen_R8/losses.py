import lpips
import torch
from utils import MulticoilAdjointOp, AdjointOp
import numpy as np
import torch.distributed as dist
import lpips


class CriterionBase(torch.nn.Module):
    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.loss_list = []
        self.use_weighting_mask = config.use_weighting_mask
        self.cardiac_crop_quantitative_metric = config.cardiac_crop_quantitative_metric

        self.k_loss_list = config.k_recon_loss_combined.k_loss_list
        self.k_loss_weighting = [1] + [config.k_recon_loss_combined.k_loss_decay ** (len(self.k_loss_list) - i - 2) for i in range(len(self.k_loss_list)-1)]  # weighting of kmae should be one
        assert len(self.k_loss_weighting) == len(self.k_loss_list)
        self.k_loss_weighting = [i*j for i, j in zip(self.k_loss_weighting, config.k_recon_loss_combined.k_loss_weighting)]

        for loss_name in config.which:
            loss_args = eval(f'config.{loss_name}').__dict__
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric' or loss_name == 'k_recon_loss':
            return PhotometricLoss(**args_dict)
        elif loss_name == 'k_recon_loss_combined':
            k_recon_loss_list = []
            for k_loss in self.k_loss_list:
                if k_loss == 'L1':
                    k_recon_loss_list.append(torch.nn.L1Loss())
                elif k_loss == 'HDR':
                    k_recon_loss_list.append(HDRLoss(eps=args_dict['eps']))
                else:
                    raise NotImplementedError
            return k_recon_loss_list
        elif loss_name == 'HDR':
            return HDRLoss(**args_dict)
        elif loss_name == 'psnr':
            return PSNR(**args_dict)
        elif loss_name == 'LPIPS':
            return LPIPSLoss(**args_dict)
        elif loss_name == 'uniformity':
            return uniformity_loss
        elif loss_name == 'KLPIPS':
            return KLPIPSLoss(**args_dict)
        else:
            raise NotImplementedError


class HDRLoss(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, input, target, weights=None, reduce=True):
        if not input.is_complex():
            input = torch.view_as_complex(input)
        if not target.is_complex():
            target = torch.view_as_complex(target)
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        # dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        # filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        # input = torch.view_as_complex(input) #* filter_value
        # target = torch.view_as_complex(target)
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        # reg_error = (input - input * filter_value)
        # reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()
        return loss.mean()


class CriterionKMAE(CriterionBase, torch.nn.Module):
    def __init__(self, config):
        super(CriterionKMAE, self).__init__(config)
        self.only_maskout = config.only_maskout


    def forward(self, k_pred, k_ref, im_pred, im_ref, kspace_mask, latent_feature=None, mask_boundary=None, mode='train'):
        total_loss = 0
        loss_dict = {}
        # kspace_mask = kspace_mask.repeat_interleave(kspace_mask.shape[2], 3)[0]
        # kspace_mask = kspace_mask.squeeze()[None, None]
        if mode == 'train': assert len(k_pred) == len(self.k_loss_list)

        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            if loss_name == 'k_recon_loss_combined':
                loss_dict[loss_name] = 0
                for pred, k_loss_term, k_loss_weights in zip(k_pred, loss_term, self.k_loss_weighting):
                    k_loss = k_loss_term(pred, k_ref)
                    loss_dict[loss_name] += k_loss_weights * k_loss
            elif loss_name == 'k_recon_loss' or loss_name == 'HDR':
                # if self.only_maskout:
                #     if not k_pred.is_complex():
                #         kspace_mask = kspace_mask.unsqueeze(-1).expand_as(k_pred)
                #     k_pred = k_pred[kspace_mask == 0]
                #     k_ref = k_ref[kspace_mask == 0]
                loss = loss_term(k_pred[0], k_ref)
                loss_dict[loss_name] = loss_weight * loss
            # elif loss_name == 'uniformity':
            #     if latent_feature is not None:
            #         loss = loss_term(latent_feature)
            #         loss_dict[loss_name] = loss_weight * loss
            else:
                loss = loss_term(im_pred, im_ref)
                loss_dict[loss_name] = loss_weight * loss
        return loss_dict


class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode):
        super(PhotometricLoss, self).__init__()
        assert mode in ('charbonnier', 'L1', 'L2', 'HDR')
        if mode == 'charbonnier':
            self.loss = CharbonnierLoss()
        elif mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'KspaceL1':
            raise NotImplementedError('KspaceL1 is not implemented yet')
        elif mode == 'L2':
            self.loss = CharbonnierLoss(eps=1.e-6, alpha=1)
        elif mode == 'HDR':
            self.loss = HDRLoss(eps=1.e-3)

    def forward(self, inputs, outputs):
        return self.loss(inputs, outputs)


def uniformity_loss(features):
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T
    eye_mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    loss = (sim.pow(2) * eye_mask).sum() / (eye_mask.sum())
    return loss


class KLPIPSLoss(torch.nn.Module):
    def __init__(self, net_type='vgg', data_convert=True, detach=True):
        super(KLPIPSLoss, self).__init__()
        self.loss = lpips.LPIPS(net=net_type).cuda()
        self.data_convert = data_convert
        self.detach = detach

    def forward(self, inputs, outputs):
        if self.data_convert:
            inputs = self.data_preprocess(inputs)
            outputs = self.data_preprocess(outputs)
        loss = self.loss(inputs, outputs)
        if self.detach:
            loss = loss.detach()
        return loss.mean()

    def data_preprocess(self, image_ch1):
        if not image_ch1.is_complex():
            image_ch1 = torch.view_as_complex(image_ch1)
        image_ch1 = torch.log(image_ch1.abs() + 0.0001)
        image_ch1 = (image_ch1 - torch.min(image_ch1)) / (torch.max(image_ch1) - torch.min(image_ch1))
        image_ch1 = image_ch1*2-1
        image_ch3 = torch.cat([image_ch1, image_ch1, image_ch1], dim=0)
        image_ch3 = image_ch3.permute(1, 0, 2, 3)
        return image_ch3


class LPIPSLoss(torch.nn.Module):
    def __init__(self, net_type='alex', data_convert=True, detach=True):
        super(LPIPSLoss, self).__init__()
        self.loss = lpips.LPIPS(net=net_type).cuda()
        self.data_convert = data_convert
        self.detach = detach

    def forward(self, inputs, outputs):
        if self.data_convert:
            inputs = self.data_preprocess(inputs)
            outputs = self.data_preprocess(outputs)
        loss = self.loss(inputs, outputs)
        if self.detach:
            loss = loss.detach()
        return loss.sum()

    def data_preprocess(self, image_ch1):
        image_ch1 = image_ch1.abs()
        image_ch1 = image_ch1/torch.max(image_ch1)*2-1
        image_ch3 = torch.cat([image_ch1, image_ch1, image_ch1], dim=0)
        image_ch3 = image_ch3.permute(1, 0, 2, 3)
        return image_ch3


class PSNR(torch.nn.Module):
    def __init__(self, max_value=1.0, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.max_value = max_value
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g):
        """

        :param u: noised image
        :param g: ground-truth image
        :param max_value:
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        square = torch.conj(diff) * diff
        max_value = g.abs().max() if self.max_value == 'on_fly' else self.max_value
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, alpha=0.45):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        diff = x - y
        square = torch.conj(diff) * diff
        if square.is_complex():
            square = square.real
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.pow(square + self.eps, exponent=self.alpha))
        return loss
