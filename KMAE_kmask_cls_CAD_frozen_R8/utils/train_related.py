import os
import glob
import torch
import wandb
import numpy as np
from datetime import datetime
import math
from utils import image_normalization, cal_lstsq_error, fix_dict_in_wandb_config


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
# def adjust_learning_rate(optimizer, epoch, args):
#     lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     return lr



def error_map_preprocess_wandb(error_map, eval_error_map_vmax):
    error_map[np.where(error_map > eval_error_map_vmax)] = eval_error_map_vmax
    if np.ndim(error_map) == 3:
        error_map[0, 0, 0] = eval_error_map_vmax
    elif np.ndim(error_map) == 4:
        error_map[0, 0, 0, 0] = eval_error_map_vmax
    elif np.ndim(error_map) == 2:
        error_map[0, 0] = eval_error_map_vmax
    else:
        raise NotImplementedError
    error_map = error_map / eval_error_map_vmax * 255
    return error_map.astype(np.uint8)


class Logger:
    def __init__(self,
                 best_eval_mode='max',
                 vis_log_video_fps=1,
                 eval_error_map_vmax=0.01):
        self.log_metric_dict = {}
        self.log_img_items = []
        self.log_video_items = []
        self.vis_log_video_fps = vis_log_video_fps
        assert best_eval_mode in ['max', 'min']
        self.best_eval_mode = best_eval_mode
        self.best_eval_result = -np.inf if best_eval_mode == 'max' else np.inf
        self.best_update_flag = False
        self.eval_error_map_vmax = eval_error_map_vmax
        # for vis_item in self.log_vis_items:
        #     exec(f'self.{vis_item} =dict()')

    def update_metric_item(self, item, value):
        if item not in self.log_metric_dict:
            self.log_metric_dict[item] = value
        else:
            self.log_metric_dict[item] += value

    # def wandb_vis_register(self, ref, recon_im, mask_boundary, name, kspace_ref, kspace, frames=8):
    #     # todo 1: make the preprocessing out of this function
    #     # todo 2: error map max?
    #     # todo 3: add kspace vis
    #     # todo: the code here is hard-coded, need to be modified!
    #     ref_vis = image_normalization(ref.abs().cpu().numpy().transpose(1, 0, 2, 3), 255, mode='3D').astype(np.uint8)
    #     ref_vis_xy = ref_vis[:frames]
    #     ref_vis_yt = ref_vis[:, :, :, int((mask_boundary[2]+mask_boundary[3])/2)].transpose(2,0,1)
    #     recon_vis = image_normalization(recon_im.abs().cpu().numpy().transpose(1, 0, 2, 3), 255, mode='3D').astype(np.uint8)
    #     recon_xy = recon_vis[:frames]
    #     recon_yt = recon_vis[:, :, :, int((mask_boundary[2]+mask_boundary[3])/2)].transpose(2,0,1)
    #     error_map = cal_lstsq_error(ref.cpu().numpy()[:, :frames, ...].transpose(1, 0, 2, 3), recon_im.cpu().numpy()[:, :frames, ...].transpose(1, 0, 2, 3))
    #    error_map = error_map_preprocess_wandb(error_map, self.eval_error_map_vmax)
    #     self.update_vis_item('recon_xy', name, wandb.Video(recon_xy, caption=name, fps=1))
    #     self.update_vis_item('ref_xy', name, wandb.Video(ref_vis_xy, caption=name, fps=1))
    #     self.update_vis_item('recon_yt', name, wandb.Image(recon_yt, caption=name))
    #     self.update_vis_item('ref_yt', name, wandb.Image(ref_vis_yt, caption=name))
    #     self.update_vis_item('error_xy', name, wandb.Video(error_map, caption=name, fps=1))
    #     # for k-space

    def wandb_log(self, epoch):
        for item in self.log_img_items:
            wandb.log({item: list(eval(f'self.{item}.values()'))}, commit=False)
        for item in self.log_video_items:
            wandb.log({item: list(eval(f'self.{item}.values()'))}, commit=False)
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(self.log_metric_dict, commit=False)
        wandb.log({'best_eval_results': self.best_eval_result})

    def wandb_log_final(self):
        test_table = wandb.Table(data=self.wandb_infer.data_list, columns=self.wandb_infer.save_table_column)
        wandb.log({'test_table': test_table}, commit=False)

    # def update_epoch(self):
    #     self.epoch += 1

    def get_metric_value(self, item):
        # print("log_metric_dic",self.log_metric_dict)
        return self.log_metric_dict[item]

    def update_img_item(self, vis_item, subj_name, value):
        if vis_item not in self.log_img_items:
            self.log_img_items.append(vis_item)
            exec(f'self.{vis_item} =dict()')
        eval(f'self.{vis_item}')[subj_name] = wandb.Image(value, caption=subj_name)

    def update_video_item(self, vis_item, subj_name, value):
        if vis_item not in self.log_video_items:
            self.log_video_items.append(vis_item)
            exec(f'self.{vis_item} =dict()')
        eval(f'self.{vis_item}')[subj_name] = wandb.Video(value, caption=subj_name, fps=self.vis_log_video_fps)

    def reset_metric_item(self):
        self.log_metric_dict = dict.fromkeys(self.log_metric_dict, 0)
        self.best_update_flag = False

    def update_best_eval_results(self, currrent_eval_result):
        sign = -1 if self.best_eval_mode == 'max' else 1
        if sign * currrent_eval_result < sign * self.best_eval_result:
            self.best_eval_result = currrent_eval_result
            self.best_update_flag = True


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def wandb_setup(args):
    if args['general']['infer_log']:
        group = args['network']['which'] + '_infer'
    else:
        group = args['network']['which']
    run = wandb.init(project='KMAE', entity=args['general']['wandb_entity'], group=group, config=args)
    group_id = args['general']['exp_name']
    wandb.config.update({'group_id' : f"{group_id}"})
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    wandb.run.name = group_id + '_' + time_now
    # wandb.run.save()
    fix_dict_in_wandb_config(wandb)
    # config = wandb.config
    # print(config)


def restore_training(model, optimizer, scheduler, args):
    if os.path.isdir(args.restore_ckpt):
        args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
    ckpt = torch.load(args.restore_ckpt)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if args.restore_training:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        args.start_epoch = ckpt['epoch']


class Scheduler:
    def __init__(self, config, optimizer, total_step):
        self.scheduler_name = config.scheduler.which
        self.update_mode = self._get_update_mode()
        self.scheduler = self._build_scheduler(optimizer, eval(f'config.scheduler.{config.scheduler.which}').__dict__, total_step)
        self.current_epoch = 0
        self.current_iter = 0

    def _get_update_mode(self):
        if self.scheduler_name in ['OneCycleLR', 'CyclicLR']:
            return 'after_iter'
        elif self.scheduler_name in ['MultiStepLR', 'CosineAnnealingLR', 'LambdaLR']:
            return 'after_epoch'
        else:
            raise NotImplementedError(f'please specify the update mode for {self.scheduler_name}.')

    def _build_scheduler(self, optimizer, args_dict, total_steps):
        if self.scheduler_name in ['MultiStepLR', 'CosineAnnealingLR']:
            return eval(f'torch.optim.lr_scheduler.{self.scheduler_name}')(optimizer, **args_dict)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=total_steps, **args_dict)
        # elif self.scheduler_name == 'CyclicLR':
        #     return torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=5*steps_per_iteration, **args_dict)
        elif self.scheduler_name == 'LambdaLR':
            # this is imitated from OneCycleLR
            lambda1 = lambda x: 0.48*x+0.04 if x<=2 else (-4/235*x+247/235 if x<=50 else 0.2)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        else:
            NotImplementedError(f'please specify the build mode for {self.scheduler_name}.')

    def update(self, mode):
        if mode == 'iter':
            if self.update_mode == 'after_iter':
                self.scheduler.step()
            self.current_iter += 1
        elif mode == 'epoch':
            if self.update_mode == 'after_epoch':
                self.scheduler.step()
            self.current_epoch += 1
            self.current_iter = 0




