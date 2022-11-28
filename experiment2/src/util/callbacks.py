import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset
from tqdm import tqdm

from . import metrics


class Evaluate:
    def __init__(self, root: str, period: int, dataset: Dataset,
                 net: nn.Module, input_shape: Tuple[int, int], num_classes: int,
                 use_cuda: bool):
        self.root = root
        self.period = period
        self.dataset = dataset
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_cuda = use_cuda

        self.x = []
        self.y = []

    def execute(self, epoch: int):
        if self.period <= 0 or epoch % self.period != 0:
            return

        # noinspection PyTypeChecker
        size = len(self.dataset)
        truths = []
        predicts = []

        dataset = [(index, self.dataset[index]) for index in range(size)]
        for index, batch in tqdm(dataset, desc='Prepare mIoU'):
            image, target = batch
            truths.append(target)

            image = np.expand_dims(image, 0)
            with torch.no_grad():
                image = torch.from_numpy(image).to(dtype=torch.float)
                if self.use_cuda:
                    image = image.cuda()

                outputs = self.net(image)[0]
                outputs = functional.softmax(outputs.permute(1, 2, 0), dim=-1).cpu().numpy()
                outputs = outputs.argmax(axis=-1)
            image = np.uint8(outputs)
            predicts.append(image)
        hist, ious, pa_recall, _ = metrics.evaluate(self.num_classes, predicts, truths)

        miou = np.nanmean(ious) * 100
        self.x.append(epoch)
        self.y.append(miou)
        with open(os.path.join(self.root, 'mIoU.txt'), 'a') as file:
            file.write(str(miou) + '\n')

        plt.figure()
        plt.plot(self.x, self.y, 'red', linewidth=2, label='Train mIoU')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('mIoU Curve')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.root, 'mIoU.png'))
        plt.cla()
        plt.close('all')

        print('mIoU: {:.2%}, mPA: {:.2%}, Accuracy: {:.2%}'.format(
            np.nanmean(ious),
            np.nanmean(pa_recall),
            metrics.per_accuracy(hist)
        ))
