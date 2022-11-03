import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, Resize, InterpolationMode, Pad, Lambda

from model import ResNet18

width, height = (500, 500)


def augmentation(image: Image.Image, target: Image.Image):
    w, h = image.size
    if w > h:
        w, h = width, int(h * (w / width))
    else:
        w, h = int(w * (h / height)), height

    source = Compose([
        Resize((h, w)),
        Pad((0, 0, max(width - w, 0), max(height - h, 0))),
        Lambda(lambda x: np.array(x))
    ])
    label = Compose([
        Resize((h, w), interpolation=InterpolationMode.NEAREST),
        Pad((0, 0, max(width - w, 0), max(height - h, 0))),
        Lambda(lambda x: np.array(x))
    ])
    return source(image), label(target)


datasets_root = os.path.join('.', 'datasets')
exist = os.path.exists(os.path.join(datasets_root, 'VOCdevkit'))
train_set = VOCSegmentation(datasets_root, image_set='train', download=not exist, transforms=augmentation)
val_set = VOCSegmentation(datasets_root, image_set='val', download=not exist, transforms=augmentation)
print(f'训练集: {len(train_set)} 张图片')
print(f'验证集: {len(val_set)} 张图片')

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

if __name__ == '__main__':
    net = ResNet18(19)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    # summary(model, (3, 2048, 1024))

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    def epoch(epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0
        for index, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if index % 1000 == 999:
                last_loss = running_loss / 1000
                print(f'  Batch #{index + 1} loss: {last_loss}')
                tb_x = epoch_index * len(train_loader) + index + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.0
        return last_loss


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    EPOCH = 5

    best_val_loss = 1_000_000.
    for current_epoch in range(EPOCH):
        print(f'EPOCH {current_epoch + 1}')

        model.train(True)
        avg_loss = epoch(current_epoch, writer)
        model.train(False)

        running_val_loss = 0.0
        for val_index, val_data in enumerate(val_loader):
            val_inputs, val_labels = val_data
            val_outputs = model(val_inputs)
            val_loss = loss_func(val_outputs, val_labels)
            running_val_loss += val_loss

        avg_val_loss = running_val_loss / (val_index + 1)
        print(f'LOSS train {avg_loss} valid {avg_val_loss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_val_loss},
                           current_epoch + 1)
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_vloss = avg_val_loss
            model_path = 'model_{}_{}'.format(timestamp, current_epoch)
            torch.save(model.state_dict(), model_path)
