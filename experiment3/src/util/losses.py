from typing import Union

import torch
from torch import FloatTensor
from torch.autograd import Variable


def scale(embedding: torch.Tensor, device: Union[torch.device, None] = None):
    one = Variable(FloatTensor([1.0]))
    zero = Variable(FloatTensor([0.0]))
    if device is not None:
        one = one.to(device=device)
        zero = zero.to(device=device)
    return torch.sum(torch.max(torch.sum(embedding ** 2, dim=1, keepdim=True) - one, zero))
