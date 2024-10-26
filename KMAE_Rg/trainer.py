import os
import sys
import pathlib
import numpy as np
import torch
import glob
import tqdm
import time
from torch.utils.data import DataLoader
from dataset.dataloader import UKBioBankKMAE
from model.kmae_2ch import KMAEDownstream
from losses import CriterionKMAE
from infer_log import InferenceLog
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, \
    NativeScalerWithGradNormCount as NativeScaler, add_weight_decay, image_normalization, \
    error_map_preprocess_wandb, cartesian_mask_yt

def calculate_accuracy(predictions, targets):
    # Assuming predictions are class probabilities (e.g., output of softmax)
    predicted_labels = predictions.argmax(dim=1)
    correct_predictions = (predicted_labels == targets).sum().item()
    total_predictions = len(targets)
    accuracy = correct_predictions / total_predictions
    print(correct_predictions )
    print(total_predictions)
    return accuracy

class TrainerAbstract:
    def __init__(self, config):
        super().__init__()
        self.config = config.general
        self.debug = config.general.debug
        if self.debug: config.general.exp_name = 'test'
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.infer_log = config.general.infer_log
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1
        self.eval_error_map_vmax = config.training.eval_error_map_vmax

        # data
        data_list = [x for x in os.listdir(config.data.data_root) if os.path.isdir(os.path.join(config.data.data_root, x))]
        self.rg_class=config.data.rg_class
        if self.rg_class=='age':
            with open('health2660.txt', 'r') as f:
                lines = f.readlines()
            final_order = [line.strip() for line in lines]
            order_dict = {item: index for index, item in enumerate(final_order)}
            data_list.sort(key=lambda x: order_dict.get(x, float('inf')))
            train_list = data_list[:1800]
            test_list = data_list[1800:2000]
        else:
            with open('filter_LVEDV_LVEF.txt', 'r') as f:
                lines = f.readlines()
            final_order = [line.strip() for line in lines]
            order_dict = {item: index for index, item in enumerate(final_order)}
            data_list.sort(key=lambda x: order_dict.get(x, float('inf')))
            train_list = data_list[:900]
            test_list = data_list[900:1000]


        train_ds = eval(f'{config.data.dataloader}')(config=config.data, mode='train', subject_list=train_list)
        test_ds = eval(f'{config.data.dataloader}')(config=config.data, mode='val', subject_list=test_list)
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                       pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=0, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        # self.network.type(torch.complex64)
        self.network.initialize_weights()
        print("Parameter Count: %d" % count_parameters(self.network))

        # optimizer
        param_groups = add_weight_decay(self.network, config.training.optim_weight_decay)
        self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(param_groups, **eval(f'config.optimizer.{config.optimizer.which}').__dict__)

        if config.training.restore_ckpt: self.load_model(config.training)
        self.loss_scaler = NativeScaler()

        if self.config.use_slurm:
            os.environ['MASTER_ADDR'] = os.environ['SLURM_NODELIST'].split(',')[0]
            os.environ['MASTER_PORT'] = '22222'
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NPROCS'])
            dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
            self.network.to(rank)
            self.network = DDP(self.network, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        self.network.cuda()

    def load_model(self, args):

        if os.path.isdir(args.restore_ckpt):
            args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
        ckpt = torch.load(args.restore_ckpt)
        self.network.load_state_dict(ckpt['model'], strict=False)

        print("Resume checkpoint %s" % args.restore_ckpt)
        if args.restore_training:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            # self.loss_scaler.load_state_dict(ckpt['scaler'])
            print("With optim & sched!")

    def save_model(self, epoch):
        ckpt = {'epoch': epoch,
                'model': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scaler': self.loss_scaler.state_dict()
                }
        torch.save(ckpt, f'{self.experiment_dir}/model_{epoch+1:03d}.pth')
        # if self.logger.best_update_flag: torch.save(save_dict, f'{self.experiment_dir}/model_best.pth')


class TrainerDownstream(TrainerAbstract):
    # todo: code needs to be check and update
    def __init__(self, config):
        super().__init__(config=config)

        self.model_mode = config.network.KspaceMAE.model_mode
        self.data_mode = config.network.KspaceMAE.data_mode
        if self.model_mode == 'reconstruction':
            self.train_criterion = CriterionKMAE(config.train_loss)
            self.eval_criterion = CriterionKMAE(config.eval_loss)
        if self.model_mode == 'classification':
            self.train_criterion = nn.CrossEntropyLoss()
            self.eval_criterion = nn.CrossEntropyLoss()
        if self.model_mode == 'regression':
            self.train_criterion = nn.SmoothL1Loss()
            self.eval_criterion = nn.SmoothL1Loss()
            #self.network.freeze_encoder_layers()

        self.eval_vis_subj = dict(zip(config.training.eval_subj, config.training.eval_slc))

        self.logger = Logger(eval_error_map_vmax=config.training.eval_error_map_vmax)
        self.scheduler_info = config.scheduler

        self.random_mask = config.training.random_mask
        self.acc = config.data.acc_rate

        # To be added later
        self.wandb_infer_log = InferenceLog(save_table_column=config.eval_log.eval_key,
                                            save_img=config.eval_log.save_img,
                                            save_dir=os.path.join(config.eval_log.save_dir, f'R{config.general.acc_rate[0]}', config.network.which),
                                            save_local=config.eval_log.save_local,
                                            save_local_item=config.eval_log.save_local_item,
                                            im_crop_size=config.eval_log.im_crop_size,
                                            save_mean_std=config.eval_log.save_mean_std)

    def run(self):
        pbar = tqdm.tqdm(range(self.start_epoch, self.num_epochs))
        for epoch in pbar:
            self.logger.reset_metric_item()
            start_time = time.time()
            if not self.only_infer:
                self.train_one_epoch(epoch)
            self.run_test()

            self.logger.update_metric_item('train/epoch_runtime', (time.time() - start_time)/60)
            # if (epoch % self.config.weights_save_frequency == 0 or self.logger.best_update_flag) and not self.debug and epoch > 150:
            if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 10:
                self.save_model(epoch)
            if epoch == self.num_epochs - 1:
                self.save_model(epoch)
            if not self.debug:
                self.logger.wandb_log(epoch)
            # if self.infer_log and not self.debug:
            if self.infer_log:
                self.wandb_infer_log.log_mean_std()
                self.wandb_infer_log.log_table()

    def train_one_epoch(self, epoch):
        self.network.train()
        for i, batch in enumerate(self.train_loader):
            kspace, sup_mask, seg = [item.cuda() for item in batch[0][0]][:]
            kspace = kspace[0]
            sup_mask = sup_mask[0]
            ref = batch[0][1][0].cuda()
            label = batch[3].cuda()
            if self.data_mode == 'image':
                img=batch[4].cuda()
                img = img.permute(0, 3, 1, 2)

            if self.random_mask:
                sup_mask = cartesian_mask_yt(kspace.shape[-1], kspace.shape[2], acc=self.acc, sample_n=5)
                sup_mask = torch.tensor(sup_mask[None, None, :, None, :].astype(np.float32)).cuda()

            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):
                if self.model_mode == 'reconstruction':
                    k_recon_2ch, im_recon = self.network(kspace, mask=sup_mask)
                    sup_mask = sup_mask[:, 0].repeat_interleave(kspace.shape[2], 2)
                    ls = self.train_criterion(k_recon_2ch, kspace, im_recon, ref,
                                              kspace_mask=sup_mask)

                    self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())
                    self.logger.update_metric_item('train/k_recon_loss',
                                                   ls['k_recon_loss'].item() / len(self.train_loader))
                    self.logger.update_metric_item('train/recon_loss',
                                                   ls['photometric'].item() / len(self.train_loader))
                if self.model_mode == 'classification':
                    k_recon_2ch = self.network(kspace, mask=sup_mask)
                    class_probs = torch.softmax(k_recon_2ch, dim=1)
                    ls = self.train_criterion(k_recon_2ch, label)
                    ls.backward()
                    self.optimizer.step()
                    self.logger.update_metric_item('train/ls',ls.item()/len(self.train_loader))
                if self.model_mode == 'regression':
                    class_label = self.network(kspace, mask=sup_mask)
                    ls = self.train_criterion(class_label, label)
                    ls.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.logger.update_metric_item('train/ls', ls.item() / len(self.train_loader))




    def run_test(self):
        self.network.eval()
        with torch.no_grad():
            class_probs_list=[]
            label_list=[]

            for i, batch in enumerate(self.test_loader):

                kspace, sup_mask, seg, = [item.cuda() for item in batch[0][0]][:]
                kspace = kspace[0]
                sup_mask = sup_mask[0]
                name, slc = batch[1][0], batch[2].item()
                ref = batch[0][1][0].cuda()
                label = batch[3].cuda()
                label_list.append(label)
                if self.data_mode == 'image':
                    img = batch[4].cuda()
                    img = img.permute(0, 3, 1, 2)


                if self.model_mode == 'reconstruction':

                    k_recon_2ch, im_recon = self.network(kspace, mask=sup_mask)
                    k_recon_2ch = k_recon_2ch[-1]
                    # kspace_complex = torch.view_as_complex(k_recon_2ch)
                    sup_mask = sup_mask[:, 0].repeat_interleave(kspace.shape[2], 2)
                    # kspace_complex[torch.where(sup_mask == 1)] = kspace[torch.where(sup_mask == 1)]

                    ls = self.eval_criterion(k_recon_2ch, kspace, im_recon, ref, kspace_mask=sup_mask, mode='test')
                    self.logger.update_metric_item('val/k_recon_loss',
                                                   ls['k_recon_loss'].item() / len(self.test_loader))
                    self.logger.update_metric_item('val/recon_loss', ls['photometric'].item() / len(self.test_loader))
                    self.logger.update_metric_item('val/psnr', ls['psnr'].item() / len(self.test_loader))

                    if self.infer_log:
                        self.wandb_infer_log.save(names=f'{self.config.exp_name}/{name}/{slc}', refs=ref,
                                                  recons=im_recon, mask_boundary=None,
                                                  eval_error_map_vmax=self.logger.eval_error_map_vmax)

                    # print(f'{name}: {np.mean(np.array([t[7] for t in self.wandb_infer_log.data_list[16*i:16*i+16]]), axis=0)}, {np.mean(np.array([t[10] for t in self.wandb_infer_log.data_list[16*i:16*i+16]]), axis=0)}')
                    if name in self.eval_vis_subj.keys() and slc == self.eval_vis_subj[name]:
                        ref_vis = image_normalization(ref.abs().cpu().numpy().transpose(1, 0, 2, 3), 255,
                                                      mode='3D').astype(np.uint8)
                        # ref_vis_yt = ref_vis[:, :, :, int((mask_boundary[2]+mask_boundary[3])/2)].transpose(2,0,1)
                        self.logger.update_video_item('ref_xy', name, ref_vis[:8])
                        # self.logger.update_img_item('ref_yt', name, ref_vis_yt)
                        recon_vis = image_normalization(im_recon.abs().cpu().numpy().transpose(1, 0, 2, 3), 255,
                                                        mode='3D').astype(np.uint8)
                        # recon_vis_yt = recon_vis[:, :, :, int((mask_boundary[2]+mask_boundary[3])/2)].transpose(2,0,1)
                        self.logger.update_video_item('recon_xy', name, recon_vis[:8])
                        # self.logger.update_img_item('recon_yt', name, recon_vis_yt)
                        # todo: need to be tested from here
                        k_recon_vis = image_normalization(
                            torch.log(k_recon_2ch.abs() + 0.0001).cpu().numpy().transpose(1, 0, 2, 3), 255,
                            mode='3D').astype(np.uint8)
                        self.logger.update_video_item('k_recon_xy', name, k_recon_vis[:8])
                        k_ref_vis = image_normalization(
                            torch.log(kspace.abs() + 0.001).cpu().numpy().transpose(1, 0, 2, 3), 255, mode='3D').astype(
                            np.uint8)
                        self.logger.update_video_item('k_ref_xy', name, k_ref_vis[:8])
                        # k_recon_vis_yt = k_recon_vis[:, :, :, int((mask_boundary[2]+mask_boundary[3])/2)].transpose(2,0,1)
                        # self.logger.update_img_item('k_recon_yt', name, k_recon_vis_yt)
                        error = (ref - im_recon).cpu().abs().numpy().transpose(1, 0, 2, 3)
                        error = error_map_preprocess_wandb(error, self.logger.eval_error_map_vmax)
                        self.logger.update_video_item('recon_error_xy', name, error[:8])
                    self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
                    self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr']/ len(self.train_loader))
                if self.model_mode == 'classification':
                    k_recon_2ch = self.network(kspace, mask=sup_mask)
                    class_probs = torch.softmax(k_recon_2ch, dim=1)
                    class_probs_list.append(class_probs)
                    ls = self.train_criterion(k_recon_2ch, label)
                    self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr']/ len(self.train_loader))
                    self.logger.update_metric_item('val/ls', ls/len(self.test_loader))
                if self.model_mode == 'regression':
                    class_label = self.network(kspace, mask=sup_mask)
                    ls = self.train_criterion(class_label, label)
                    self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr']/ len(self.train_loader))
                    self.logger.update_metric_item('val/ls', ls / len(self.test_loader))
                    self.logger.update_metric_item('val/regression_mae',
                                                   torch.abs(class_label - label).mean().item() / len(self.test_loader))


            if self.model_mode == 'classification':
                label_list=torch.cat(label_list,0)
                class_probs_list=torch.cat(class_probs_list, 0)
                accuracy = calculate_accuracy(class_probs_list, label_list)
                self.logger.update_metric_item('val/accuracy', accuracy)





