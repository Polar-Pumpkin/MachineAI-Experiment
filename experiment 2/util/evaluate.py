import numpy as np


# 得到混淆矩阵
def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 计算图像分割衡量系数
def evaluate(label_truth, label_predict, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for truth, predict in zip(label_truth, label_predict):
        hist += fast_hist(truth.flatten(), predict.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
