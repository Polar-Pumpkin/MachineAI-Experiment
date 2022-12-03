import torch
from torch import FloatTensor
from torch.autograd import Variable


def scale(embedding: torch.Tensor):
    return torch.sum(
        torch.max(
            torch.sum(embedding ** 2, dim=1, keepdim=True) - Variable(FloatTensor([1.0])),
            Variable(FloatTensor([0.0]))
        )
    )
