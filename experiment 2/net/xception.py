import math
from typing import Union, List

import torch
import torch.nn as nn
import torch.serialization as serialization

from ..util import models


class SeparableConv2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, momentum: float = 0.0003,
                 bias: bool = False, activate_first: bool = True, inplace: bool = True):
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation,
                                   groups=input_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(input_channels, momentum=momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, input_filters: int, output_filters: int,
                 strides: int = 1, atrous: Union[List[int], int, None] = None, momentum=0.0003,
                 grow_first: bool = True, activate_first: bool = True, inplace: bool = True):
        super(Block, self).__init__()
        if atrous is None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous = [atrous] * 3

        self.head_relu = True
        if output_filters != input_filters or strides != 1:
            self.skip = nn.Conv2d(input_filters, output_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(output_filters, momentum=momentum)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None
        if grow_first:
            filters = output_filters
        else:
            filters = input_filters

        self.sepconv1 = SeparableConv2d(input_filters, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0],
                                        bias=False, activate_first=activate_first, inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters, output_filters, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1],
                                        bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(output_filters, output_filters, 3, stride=strides, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first, inplace=inplace)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        output = self.sepconv1(x)
        output = self.sepconv2(output)
        self.hook_layer = output
        output = self.sepconv3(output)

        output += skip
        return output


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, downsample_factor: int, momentum: float = 0.0003):
        super(Xception, self).__init__()
        _strides = {
            8: [2, 1, 1],
            16: [2, 2, 1]
        }
        strides = _strides[downsample_factor]
        assert strides is not None, '不支持该下采样系数: {}'.format(downsample_factor)

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=momentum)
        # do relu here

        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, strides[0], inplace=False)
        self.block3 = Block(256, 728, strides[1])

        rate = 16 // downsample_factor
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)

        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)

        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)

        self.block16 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block17 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block18 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block19 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

        self.block20 = Block(728, 1024, strides[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, inputs):
        self.layers = []
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return low_featrue_layer, x

    @staticmethod
    def pretrained(downsample_factor: int = 16, map_location: serialization.MAP_LOCATION = None, **kwargs):
        url = 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet' \
              '.pth '
        return models.pretrained(Xception, url, map_location, downsample_factor=downsample_factor, **kwargs)


if __name__ == '__main__':
    model = Xception.pretrained(map_location=torch.device('cpu'))
    for index, layer in enumerate(model.features):
        print(index, layer)
