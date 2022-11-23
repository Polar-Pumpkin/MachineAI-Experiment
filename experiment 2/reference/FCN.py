# #Phase 1

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# # ResNet基础模块
# class RestNetBasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(RestNetBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         output = F.relu(self.bn1(output))
#         output = self.conv2(output)
#         output = self.bn2(output)
#         return F.relu(x + output)
#
# # ResNet下采样模块
# class RestNetDownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(RestNetDownBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.extra = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
#             nn.BatchNorm2d(out_channels)
#         )
#
#     def forward(self, x):
#         extra_x = self.extra(x)
#         output = self.conv1(x)
#         out = F.relu(self.bn1(output))
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         return F.relu(extra_x + out)
#
#
# class RestNet18(nn.Module):
#     def __init__(self):
#         super(RestNet18, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
#                                     RestNetBasicBlock(64, 64, 1))
#
#         self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
#                                     RestNetBasicBlock(128, 128, 1))
#
#         self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
#                                     RestNetBasicBlock(256, 256, 1))
#
#         self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
#                                     RestNetBasicBlock(512, 512, 1))
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         self.fc = nn.Linear(512, 10)
#         # self.softmax=nn.Softmax()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # out = self.avgpool(out)
#         # out = out.reshape(x.shape[0], -1)
#         # out = self.fc(out)
#
#         return out
#
# net=RestNet18()
# X=torch.rand(1,3,224,224)
# print(net(X))


#Phase 2
# import torch
# import torchvision
# from torch import nn
# from torch.nn import functional as F
# from d2l import torch as d2l
#
# pretrained_net=torchvision.models.resnet18(pretrained=True)
# # print(list(pretrained_net.children())[-3:])
# net=nn.Sequential(*list(pretrained_net.children())[:-2])
# X=torch.rand(1,3,224,224)
# # print(net(X).shape)
# #
# num_classes=4
# net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
# net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))
# print(net(X).shape)
# from torchsummary import summary
#
# if __name__ == "__main__":
#     input_shape = [224, 224]
#     num_classes = 4
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = net.to(device)
#     summary(model, (3, input_shape[0], input_shape[1]))