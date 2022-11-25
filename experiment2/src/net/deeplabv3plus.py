from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .mobilenetv2 import MobileNetV2
from .xception import Xception


# ASPP 特征提取模块
# 利用不同膨胀率的膨胀卷积进行特征提取
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int = 1, momentum: float = 0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [_, _, row, col] = x.size()
        # 一共五个分支
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 第五个分支, 全局平均池化 + 卷积
        b5 = self.branch5(x)
        b5 = functional.interpolate(b5, (row, col), None, 'bilinear', True)

        # 将五个分支的内容堆叠起来
        # 然后 1x1 卷积整合特征
        feature_cat = torch.cat([b1, b2, b3, b4, b5], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class _MobileNetV2(nn.Module):
    def __init__(self, downsample_factor: int = 8, pretrained: bool = True, device=None):
        super(_MobileNetV2, self).__init__()

        model = MobileNetV2.pretrained(device) if pretrained else MobileNetV2()
        self.features = model.features[:-1]

        self.total_indexs = len(self.features)
        self.down_indexs = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_indexs[-2], self.down_indexs[-1]):
                self.features[i].apply(partial(self._set_dilate, dilate=2))
            for i in range(self.down_indexs[-1], self.total_indexs):
                self.features[i].apply(partial(self._set_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_indexs[-1], self.total_indexs):
                self.features[i].apply(partial(self._set_dilate, dilate=2))

    @staticmethod
    def _set_dilate(module, dilate):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            if module.stride == (2, 2):
                module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilate // 2, dilate // 2)
                    module.padding = (dilate // 2, dilate // 2)
            else:
                if module.kernel_size == (3, 3):
                    module.dilation = (dilate, dilate)
                    module.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int, backbone: str = 'mobilenet', downsample_factor: int = 16,
                 pretrained: bool = True):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'xception':
            # 获得两个特征层
            # 浅层特征 [128, 128, 256]
            # 主干部分 [30, 30, 2048]
            self.backbone = Xception.pretrained(downsample_factor) if pretrained else Xception()
            in_channels = 2048
            low_level_channels = 256
        elif backbone == 'mobilenet':
            # 获得两个特征层
            # 浅层特征 [128, 128, 24]
            # 主干部分 [30, 30, 320]
            self.backbone = _MobileNetV2(downsample_factor, pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone: `{}` (xception, mobilenet)'.format(backbone))

        # ASPP特征提取模块
        # 利用不同膨胀率的膨胀卷积进行特征提取
        self.aspp = ASPP(in_channels=in_channels, out_channels=256, rate=16 // downsample_factor)

        # 浅层特征边
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        print('before backbone:', x.size())
        height, width = x.size(2), x.size(3)
        # 获得两个特征层
        # low_level_features: 浅层特征, 进行卷积处理
        # x: 主干部分, 利用 ASPP 结构进行加强特征提取
        low_level_features, x = self.backbone(x)
        print('after backbone:', x.size())
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # 将加强特征边上采样
        # 与浅层特征堆叠后利用卷积进行特征提取
        x = functional.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                                   align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = functional.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        return x
