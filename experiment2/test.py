import torch

from experiment2.src.net import DeepLabV3Plus

if __name__ == '__main__':
    net = DeepLabV3Plus(21, device=torch.device('cpu'))
    net.train()
    X = torch.randn(4, 3, 512, 512)
    y = net(X)
    print(type(y))
    print(y)
