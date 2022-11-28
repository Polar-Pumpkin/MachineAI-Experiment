import numpy as np

import torch
import torch.nn.functional as functional
from tqdm import tqdm


# noinspection DuplicatedCode,PyArgumentList
def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    _inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    _target = target.view(n, -1, ct)

    _inputs = torch.gt(_inputs, threhold).float()
    tp = torch.sum(_target[..., :-1] * _inputs, axis=[0, 1])
    fp = torch.sum(_inputs, axis=[0, 1]) - tp
    fn = torch.sum(_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return torch.mean(score)


# 设标签宽W，长H
def fast_hist(a, b, n: int):
    # a 是转化成一维数组的标签, 形状 (H×W,); b 是转化成一维数组的预测结果, 形状 (H×W,)
    k = (a >= 0) & (a < n)
    # np.bincount 计算了从 0 到 n**2-1 这 n**2 个数中每个数出现的次数, 返回值形状 (n, n)
    # 返回中, 写对角线上的为分类正确的像素点
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_pa_recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def evaluate(num_classes: int, predicts: list, truths: list):
    size = min(len(predicts), len(truths))
    hist = np.zeros((num_classes, num_classes))
    for index in tqdm(range(size), desc='Evaluate mIoU'):
        predict = predicts[index]
        truth = truths[index]

        if len(truth.flatten()) != len(predict.flatten()):
            print('跳过 #{}: {} -> {}'.format(index, len(predict.flatten()), len(truth.flatten())))
            continue
        hist += fast_hist(truth.flatten(), predict.flatten(), num_classes)

    ious = per_class_iu(hist)
    pa_recall = per_class_pa_recall(hist)
    precision = per_class_precision(hist)
    return hist, ious, pa_recall, precision

