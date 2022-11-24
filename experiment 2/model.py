import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(BasicBlock(64, 64, 1),
                                    BasicBlock(64, 64, 1))
        self.stage2 = nn.Sequential(DownsampleBlock(64, 128, [2, 1]),
                                    BasicBlock(128, 128, 1))
        self.stage3 = nn.Sequential(DownsampleBlock(128, 256, [2, 1]),
                                    BasicBlock(256, 256, 1))
        self.stage4 = nn.Sequential(DownsampleBlock(256, 512, [2, 1]),
                                    BasicBlock(512, 512, 1))
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)
        self.dot = nn.Conv2d(512, num_classes, kernel_size=1)

    # @staticmethod
    # def debug(**kwargs):
    #     for name, value in kwargs.items():
    #         if not isinstance(value, torch.Tensor):
    #             print(f'{name}: is not a Tensor')
    #         print(f'{name}: Requires grad: {value.requires_grad}, Grad function: {value.grad_fn}')

    def forward(self, x):
        output = self.conv1(x)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.upsample(output)
        output = self.dot(output)
        # self.debug(x=x, output=output)
        return output


if __name__ == '__main__':
    net = ResNet18(21)
    X = torch.rand(1, 3, 512, 512)
    y = net(X)
    print(y.shape)
