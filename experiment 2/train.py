import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import VOCSegmentation

from model import ResNet18
from util.callbacks import Evaluate
from util.losses import LossHistory
from util.train import get_lr_scheduler
from util.train import one_epoch

# 使用 Cuda
use_cuda = cuda.is_available()
# 使用分布式运行 (单机多卡)
use_distributed = False
# 全局梯度同步 (用于 DDP)
sync_bn = False
# 混合精度训练, 可减少约一半的显存 (需要 PyTorch 1.7.1+)
use_fp16 = False
# 分类数量
num_classes = 21
# 输入图片大小
input_shape = (512, 512)
width, height = input_shape
# 学习率与学习率下降
lr_init = 7e-3
lr_min = lr_init * 0.01
lr_decay_type = 'cos'
# 优化器与优化器参数
optimizer_type = 'sgd'
momentum = 0.9
weight_decay = 1e-4
# 损失函数
use_dice_loss = False
use_focal_loss = False
# 使用多线程读取数据
num_workers = 1
#
epoch_from = 0
epoch_freeze = 0
epoch_max = 50
batch_size_freeze = 8
batch_size_unfreeze = 4
freeze_train = False
#
eval_period = 1
#
class_weights = np.ones([num_classes], np.float32)
#
save_period = 5
save_folder = os.path.join('.', 'runs')
#
datasets_path = os.path.join('.', 'datasets')


def augmentation(image, target):
    w, h = image.size
    if w > h:
        w, h = width, int(h * (w / width))
    else:
        w, h = int(w * (h / height)), height

    top = 0
    bottom = max(height - h, 0)
    left = 0
    right = max(width - w, 0)

    image = np.array(image, np.float64)
    image = cv2.resize(image, (w, h))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0

    target = np.array(target)
    target = cv2.resize(target, (w, h), interpolation=cv2.INTER_NEAREST)
    target = cv2.copyMakeBorder(target, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # target = target / 255.0
    # target[target >= num_classes] = num_classes

    # labels = np.eye(num_classes + 1)[target.reshape([-1])]
    # labels = labels.reshape((width, height, num_classes + 1))
    return image, target


def collate(batch):
    imgs = []
    pngs = []
    labels = []
    for img, png in batch:
        imgs.append(img)

        png[png >= num_classes] = num_classes
        pngs.append(png)

        label = np.eye(num_classes + 1)[png.reshape([-1])]
        label = label.reshape((width, height, num_classes + 1))
        labels.append(label)
    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return imgs, pngs, labels


if __name__ == '__main__':
    ngpus_per_node = cuda.device_count()
    if use_cuda and use_distributed:
        distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        device = torch.device('cuda', local_rank)
        if local_rank == 0:
            print(f'[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...')
            print('GPU 设备数量:', ngpus_per_node)
    else:
        device = torch.device('cuda' if use_cuda and cuda.is_available() else 'cpu')
        local_rank = 0

    model = ResNet18(num_classes)
    root_path = os.path.join(save_folder, datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))

    if local_rank == 0:
        history = LossHistory(root_path, model, input_shape)
    else:
        history = None

    if use_fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and use_cuda and use_distributed and ngpus_per_node > 1:
        model_train = nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print('无法在单 GPU 设备或非分布式训练的情况下全局同步梯度')

    if use_cuda:
        if use_distributed:
            model_train = model_train.cuda(local_rank)
            model_train = nn.parallel.DistributedDataParallel(model_train,
                                                              device_ids=[local_rank],
                                                              find_unused_parameters=True)
        else:
            model_train = nn.DataParallel(model)
            cudnn.benchmark = False
            model_train = model_train.cuda()

    exist = os.path.exists(os.path.join(datasets_path, 'VOCdevkit'))
    train_set = VOCSegmentation(datasets_path, image_set='train', download=not exist, transforms=augmentation)
    train_size = len(train_set)
    validate_set = VOCSegmentation(datasets_path, image_set='val', download=not exist, transforms=augmentation)
    validate_size = len(validate_set)
    print(f'训练集: {train_size} 张图片')
    print(f'验证集: {validate_size} 张图片')

    if local_rank == 0:
        # TODO 显示配置文件
        step_wanted = 1.5e4 if optimizer_type == 'sgd' else 0.5e4
        step_total = train_size // batch_size_unfreeze * epoch_max
        if step_total <= step_wanted:
            if train_size // batch_size_unfreeze == 0:
                raise ValueError('无法进行训练: 数据集过小, 请扩充数据集')
            epoch_wanted = step_wanted // (train_size // batch_size_unfreeze) + 1
            print('使用 %s 优化器时, 建议将训练总步长设置到 %d 以上' % (optimizer_type, step_wanted))
            print('本次运行的总训练数据量为 %d, 解冻训练的批大小为 %d, 共训练 %d 轮, 计算出总训练步长为 %d' % (
                train_size, batch_size_unfreeze, epoch_max, step_total))
            print(
                '由于总训练步长为 %d, 小于建议总步长 %d, 建议设置总轮数为 %d' % (step_total, step_wanted, epoch_wanted))

    is_unfreezed = False
    if freeze_train:
        for param in model.parameters():
            param.requires_grad = False

    batch_size = batch_size_freeze if freeze_train else batch_size_unfreeze
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    lr_init_fit = min(max(batch_size / nbs * lr_init, lr_limit_min), lr_limit_max)
    lr_min_fit = min(max(batch_size / nbs * lr_min, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, lr_init_fit, lr_min_fit, epoch_max)

    optimizers = {
        'adam': optim.Adam(model.parameters(), lr_init_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr_init_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }
    optimizer = optimizers[optimizer_type]

    epoch_step = train_size // batch_size
    epoch_step_validate = validate_size // batch_size
    if epoch_step == 0 or epoch_step_validate == 0:
        raise ValueError('无法进行训练: 数据集过小, 请扩充数据集')

    if use_cuda and use_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        validate_sampler = DistributedSampler(validate_set, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        validate_sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, drop_last=True, collate_fn=collate, sampler=train_sampler)
    validate_loader = DataLoader(validate_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=collate, sampler=validate_sampler)

    if local_rank == 0:
        evaluate = Evaluate(root_path, eval_period, validate_set, model, input_shape, num_classes, use_cuda)
    else:
        evaluate = None

    for epoch in range(epoch_from, epoch_max):
        if epoch >= epoch_freeze and not is_unfreezed:
            # TODO 代码复用 (重新设置超参数)
            for param in model.parameters():
                param.requires_grad = True
            is_unfreezed = True

        if use_cuda and use_distributed:
            train_sampler.set_epoch(epoch)

        lr = lr_scheduler_func(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        one_epoch(epoch, epoch_max, model_train, model, optimizer, num_classes, class_weights, scaler,
                  train_loader, validate_loader, epoch_step, epoch_step_validate,
                  use_cuda, use_fp16, use_dice_loss, use_focal_loss,
                  history, evaluate,
                  save_period, root_path, local_rank)

        if use_cuda and use_distributed:
            distributed.barrier()

    if local_rank == 0:
        history.writer.close()
