import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distributed
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

from model import ResNet18
from util.evaluate import evaluate

width, height = (512, 512)


def augmentation(image, target):
    w, h = image.size
    if w > h:
        w, h = width, int(h * (w / width))
    else:
        w, h = int(w * (h / height)), height

    top = 0
    bottom = max(height - h, 0)
    left = 0
    right = max(width - w, 0)

    image = np.array(image)
    image = cv2.resize(image, (w, h))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0

    target = np.array(target)
    target = cv2.resize(target, (w, h), interpolation=cv2.INTER_NEAREST)
    target = cv2.copyMakeBorder(target, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    target = target / 255.0
    return image, target


datasets_root = os.path.join('.', 'datasets')
exist = os.path.exists(os.path.join(datasets_root, 'VOCdevkit'))
train_set = VOCSegmentation(datasets_root, image_set='train', download=not exist, transforms=augmentation)
val_set = VOCSegmentation(datasets_root, image_set='val', download=not exist, transforms=augmentation)
print(f'训练集: {len(train_set)} 张图片')
print(f'验证集: {len(val_set)} 张图片')

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

if __name__ == '__main__':
    net = ResNet18(21)

    distributed.init_process_group(backend="nccl")
    # device_ids will include all GPU devices by default
    model = DistributedDataParallel(net.cuda())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = net.to(device)
    # summary(model, (3, 2048, 1024))

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch = 15
    best_score = 0.0
    for epoch_index in range(epoch):
        model.train()
        train_loss = 0.0

        train_truth = torch.LongTensor()
        train_predict = torch.LongTensor()
        for batch_index, batch in enumerate(train_loader):
            inputs, labels = map(lambda x: x.to(device=device, dtype=torch.float), batch)

            outputs = functional.log_softmax(model(inputs), dim=1)
            loss = loss_func(outputs, labels)

            truth = labels.data.cpu()
            predict = outputs.argmax(dim=1).squeeze().data.cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item() * labels.size(0)
            train_truth = torch.cat((train_truth, truth), dim=0)
            train_predict = torch.cat((train_predict, predict), dim=0)

        train_loss /= len(train_set)
        acc, acc_cls, mean_iu, fwavacc = evaluate(train_truth.numpy(), train_predict.numpy(), 21)
        print('\nepoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'
              .format(epoch_index + 1, train_loss, acc, acc_cls, mean_iu, fwavacc))

        model.eval()
        validate_loss = 0.0

        validate_truth = torch.LongTensor()
        validate_predict = torch.LongTensor()
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                inputs, labels = map(lambda x: x.to(device=device, dtype=torch.float), batch)

                outputs = functional.log_softmax(model(inputs), dim=1)
                loss = loss_func(outputs, labels)

                truth = labels.data.cpu()
                predict = outputs.argmax(dim=1).squeeze().data.cpu()

                validate_loss += loss.cpu().item() * labels.size(0)
                validate_truth = torch.cat((validate_truth, truth), dim=0)
                validate_predict = torch.cat((validate_predict, predict), dim=0)

            validate_loss /= len(val_set)
            acc, acc_cls, mean_iu, fwavacc = evaluate(validate_truth.numpy(), validate_predict.numpy(), 21)
            print('\nepoch:{}, validate_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'
                  .format(epoch_index + 1, validate_loss, acc, acc_cls, mean_iu, fwavacc))

        score = (acc_cls + mean_iu) / 2
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), './runs/best_result.pth')
