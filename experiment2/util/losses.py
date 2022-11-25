import os
from typing import Tuple

import matplotlib.pyplot as plt
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter


class LossHistory:
    def __init__(self, root: str, model: nn.Module, input_shape: Tuple[int, int]):
        self.root = root
        self.losses = []
        self.validate_losses = []

        os.makedirs(root, exist_ok=True)
        self.writer = SummaryWriter(root)

        width, height = input_shape
        # noinspection PyBroadException
        try:
            self.writer.add_graph(model, torch.randn(2, 3, width, height))
        except Exception as _:
            pass

    def append(self, epoch, loss, validate_loss):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.losses.append(loss)
        self.validate_losses.append(validate_loss)

        with open(os.path.join(self.root, 'losses.txt'), 'a') as file:
            file.write(str(loss) + '\n')
        with open(os.path.join(self.root, 'validate_losses.txt'), 'a') as file:
            file.write(str(validate_loss) + '\n')

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('validate_loss', validate_loss, epoch)
        self.plot()

    def plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='Train loss')
        plt.plot(iters, self.validate_losses, 'coral', linewidth=2, label='Validate loss')
        # noinspection PyBroadException
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, signal.savgol_filter(self.losses, num, 3), 'green',
                     linestyle='--', linewidth=2, label='Smooth train loss')
            plt.plot(iters, signal.savgol_filter(self.validate_losses, num, 3), '#8B4513',
                     linestyle='--', linewidth=2, label='Smooth val loss')
        except Exception as _:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.root, 'losses.png'))
        plt.cla()
        plt.close('all')


def _prepare(inputs, target):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    _inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    _target = target.view(-1)
    return _inputs, _target


def ce(inputs, target, class_weights, num_classes=21):
    _inputs, _target = _prepare(inputs, target)
    func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=num_classes)
    return func(_inputs, _target)


def focal(inputs, target, class_weights, num_classes=21, alpha=0.5, gamma=2):
    _inputs, _target = _prepare(inputs, target)
    func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=num_classes, reduction='none')
    logpt = -func(_inputs, _target)

    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()


# noinspection DuplicatedCode,PyArgumentList
def dice(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    _inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    _target = target.view(n, -1, ct)

    tp = torch.sum(_target[..., :-1] * _inputs, axis=[0, 1])
    fp = torch.sum(_inputs, axis=[0, 1]) - tp
    fn = torch.sum(_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return 1 - torch.mean(score)
