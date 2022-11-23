import torch
import torch.nn.functional as F


# noinspection DuplicatedCode,PyArgumentList
def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    _inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    _target = target.view(n, -1, ct)

    _inputs = torch.gt(_inputs, threhold).float()
    tp = torch.sum(_target[..., :-1] * _inputs, axis=[0, 1])
    fp = torch.sum(_inputs, axis=[0, 1]) - tp
    fn = torch.sum(_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return torch.mean(score)
