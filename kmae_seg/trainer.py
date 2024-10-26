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
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice

from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, \
    NativeScalerWithGradNormCount as NativeScaler, add_weight_decay, image_normalization, \
    error_map_preprocess_wandb, cartesian_mask_yt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualize_segmentation_and_save(image, segmentation, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 获取中间一帧图像和分割结果
    mid_index = image.shape[0] // 2
    mid_image = image[mid_index]
    mid_segmentation = segmentation[mid_index]

    # 显示中间一帧图像
    axes[0].imshow(mid_image, cmap='gray')  # 使用灰度颜色映射显示黑白图像
    axes[0].set_title('Image')
    axes[0].axis('off')

    # 创建红色颜色映射
    cmap_red = LinearSegmentedColormap.from_list('blue', [(0, 'white'), (1, 'blue')])

    # 叠加显示中间一帧的分割结果，1的部分显示为红色
    axes[1].imshow(mid_image, cmap='gray')  # 使用灰度颜色映射显示黑白图像
    axes[1].imshow(mid_segmentation, cmap=cmap_red, alpha=0.5)  # 使用红色颜色映射显示分割结果
    axes[1].set_title('Segmentation')
    axes[1].axis('off')

    # 保存图像
    plt.savefig(save_path)
    plt.close()

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
        data_list = [x for x in os.listdir(config.data.data_root) if
                     os.path.isdir(os.path.join(config.data.data_root, x))]
        with open('health2660.txt', 'r') as f:
            lines = f.readlines()
        final_order = [line.strip() for line in lines]
        order_dict = {item: index for index, item in enumerate(final_order)}
        data_list.sort(key=lambda x: order_dict.get(x, float('inf')))

        # data_list.sort()

        # train_list = data_list[:min([int(config.data.data_split.split('/')[0]), len(data_list) - int(config.data.data_split.split('/')[1])])]
        # test_list = data_list[-int(config.data.data_split.split('/')[1]):]
        train_list = data_list[:9]
        test_list = data_list[9:10]


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

        # self.train_criterion = CriterionKMAE(config.train_loss)
        # self.eval_criterion = CriterionKMAE(config.eval_loss)
        self.eval_criterion = torch.nn.BCELoss()
        self.train_criterion =torch.nn.BCELoss()


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
            if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 20:
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
            ref = batch[0][1][0].cuda()

            if self.random_mask:
                sup_mask = cartesian_mask_yt(kspace.shape[-1], kspace.shape[2], acc=self.acc, sample_n=5)
                sup_mask = torch.tensor(sup_mask[None, None, :, None, :].astype(np.float32)).cuda()

            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):

                k_recon_2ch, im_recon = self.network(kspace, mask=sup_mask)
                k_recon_2ch=k_recon_2ch[0]
                k_recon_2ch = torch.view_as_complex(k_recon_2ch)
                # image_normalized = (im_recon - im_recon.min()) / (im_recon.max() - im_recon.min())
                image_normalized_sigmoid=torch.sigmoid(im_recon)
                # im_recon_sigmoid=torch.abs(im_recon_sigmoid)
                ls = self.train_criterion(image_normalized_sigmoid, seg)
                ls.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.update_metric_item('train/ls', ls.item() / len(self.train_loader))

                # self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())

            # self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
            # self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

    def run_test(self):
        self.network.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                kspace, sup_mask, seg, = [item.cuda() for item in batch[0][0]][:]
                name, slc = batch[1][0], batch[2].item()
                ref = batch[0][1][0].cuda()

                k_recon_2ch, im_recon = self.network(kspace, mask=sup_mask)
                k_recon_2ch = k_recon_2ch[0]
                k_recon_2ch = torch.view_as_complex(k_recon_2ch)
                # image_normalized = (im_recon - im_recon.min()) / (im_recon.max() - im_recon.min())
                image_normalized_sigmoid = torch.sigmoid(im_recon)
                # im_recon_sigmoid=torch.abs(im_recon_sigmoid)
                ls = self.train_criterion(image_normalized_sigmoid, seg)
                im_recon_sigmoid_binary = (image_normalized_sigmoid > 0.5).float()  # 将大于0.5的值设置为1，小于等于0.5的值设置为0
                # class_probs_list.append(k_recon_2ch_un_patch_binary)
                self.logger.update_metric_item('val/ls', ls / len(self.test_loader))
                metric = BinaryJaccardIndex().to('cuda')
                accuracy = metric(im_recon_sigmoid_binary, seg)
                self.logger.update_metric_item('val/iou_eachbatch', accuracy/len(self.test_loader))
                dice = Dice(average='micro').to('cuda')
                seg = seg.to(torch.int)
                score = dice(im_recon_sigmoid_binary, seg)
                self.logger.update_metric_item('val/dice', score / len(self.test_loader))

                image = ref[0].cpu().numpy()  # 将图像张量移动到CPU并转换为NumPy数组
                seg = im_recon_sigmoid_binary[0].cpu().numpy()  # 将分割结果张量移动到CPU并转换为NumPy数组
                save_path = "visualization_predict_KMAE_R8.png"
                visualize_segmentation_and_save(image, seg, save_path)

