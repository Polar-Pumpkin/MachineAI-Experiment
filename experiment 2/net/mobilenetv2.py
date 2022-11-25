import math

import torch
import torch.nn as nn
import torch.serialization as serialization

from ..util import models


class InvertedResidual(nn.Module):
    def __init__(self, input_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = round(input_channels * expand_ratio)
        self.use_residual_connect = self.stride == 1 and input_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # 进行 3x3 的逐层卷积，进行跨特征点的特征提取
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 利用 1x1 卷积进行通道数的调整
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # 利用 1x1 卷积进行通道数的上升
                nn.Conv2d(input_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 进行 3x3 的逐层卷积，进行跨特征点的特征提取
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 利用 1x1 卷积进行通道数的下降
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_residual_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, input_size: int = 224, width_multiple: float = 1.0):
        super(MobileNetV2, self).__init__()
        assert input_size % 32 == 0
        input_channels = int(32 * width_multiple)
        self.last_channel = int(1280 * width_multiple) if width_multiple > 1.0 else 1280
        # 512, 512, 3 -> 256, 256, 32
        self.features = [
            nn.Sequential(
                nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU6(inplace=True)
            )
        ]

        interverted_residuals = [
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2],  # 32, 32, 96 -> 16, 16, 160     14
            [6, 320, 1, 1],  # 16, 16, 160 -> 16, 16, 320
        ]
        for t, c, n, s in interverted_residuals:
            output_channels = int(c * width_multiple)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channels, output_channels, s, expand_ratio=t))
                else:
                    self.features.append(InvertedResidual(input_channels, output_channels, 1, expand_ratio=t))
                input_channels = output_channels

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                # n = module.weight.size(1)
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @staticmethod
    def pretrained(map_location: serialization.MAP_LOCATION = None, **kwargs):
        url = 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'
        return models.pretrained(MobileNetV2, url, map_location, num_classes=1000, **kwargs)


if __name__ == '__main__':
    model = MobileNetV2.pretrained(torch.device('cpu'))
    for index, layer in enumerate(model.features):
        print(index, layer)
