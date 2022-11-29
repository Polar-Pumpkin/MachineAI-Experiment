import argparse
import colorsys
import itertools
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from tqdm import tqdm

from src.net import DeepLabV3Plus
from src.util.general import fill


class AdaptedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super(AdaptedModule, self).__init__()
        self.add_module('module', module)
        self.module = module

    def forward(self, x):
        return self.module(x)


parser = argparse.ArgumentParser()

parser.add_argument('input', help='输入图像的路径')
parser.add_argument('-m', '--model', help='模型的路径')
parser.add_argument('-d', '--detail', help='输出 21 张预测图', action='store_true')

args = parser.parse_args()

input_path: str = args.input
if not os.path.exists(input_path):
    print(f'无效的输入图像路径: {input_path}')
    exit()

images: List[Tuple[str, str]] = []
if os.path.isdir(input_path):
    bar = tqdm(os.listdir(input_path), desc='Prepare Images')
    bar.set_postfix(**{'filename': '?'})
    for filename in bar:
        bar.set_postfix(**{'filename': filename})
        try:
            _, ext = os.path.splitext(filename)
            assert str(ext).lower() in ['.jpg', '.png'], f'不支持的文件类型: {filename}'
            images.append((filename, os.path.join(input_path, filename)))
        except AssertionError as ex:
            print(str(ex))
else:
    try:
        _, filename = os.path.split(input_path)
        _, ext = os.path.splitext(input_path)
        assert str(ext).lower() in ['.jpg', '.png'], f'不支持的文件类型: {filename}'
        images.append((filename, input_path))
    except AssertionError as ex:
        print(str(ex))
if not len(images) > 0:
    print(f'无效的输入图像路径: {input_path}')
    exit()
else:
    print(f'输入图像: {len(images)} 张')

model_path: str
if args.model is not None:
    model_path = args.model
else:
    latest = None
    models = [(name, os.path.join('runs', name)) for name in os.listdir('runs')]
    for name, filename in tqdm(models, desc='Prepare Model'):
        if not os.path.isdir(filename) or not os.path.exists(os.path.join(filename, 'best.pth')):
            continue

        try:
            timestamp = time.strptime(name, '%Y-%m-%d-%H-%M-%S')
        except ValueError:
            continue

        if latest is None:
            latest = name, timestamp
        else:
            _, latest_timestamp = latest
            if timestamp > latest_timestamp:
                latest = name, timestamp
    name, _ = latest
    model_path = os.path.join('runs', name, 'best.pth')
if not os.path.exists(model_path):
    print(f'无效的模型路径: {model_path}')
    exit()
else:
    print(f'模型路径: {model_path}')

for folder in ['images', 'arrays', 'masks']:
    path = os.path.join('output', folder)
    if not os.path.exists(path):
        os.makedirs(path)

input_shape = (512, 512)
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
colors = [(x / 21, 1.0, 1.0) for x in range(21)]
colors = itertools.starmap(colorsys.hsv_to_rgb, colors)
colors = map(lambda x: tuple(map(lambda y: int(y * 255), x)), colors)
colors = list(colors)
colors = np.array(colors, np.uint8)

net = DeepLabV3Plus(21, pretrained=False)
net = AdaptedModule(net)
missing, unexpected = net.load_state_dict(torch.load(model_path), False)
if len(unexpected) > 0:
    with open(os.path.join('output', 'unexpected_keys.txt'), 'w') as file:
        file.writelines(unexpected)
    print(f'未知 {len(unexpected)} Keys, 已保存至 unexpected_keys.txt')
if len(missing) > 0:
    with open(os.path.join('output', 'missing_keys.txt'), 'w') as file:
        file.writelines(missing)
    print(f'缺失 {len(missing)} Keys, 已保存至 missing_keys.txt')
    exit()
print(f'已加载模型权重')
net.eval()

predicts = []
bar = tqdm(images, desc='Evaluate')
bar.set_postfix(**{'filename': '?', 'size': '?'})
for filename, path in bar:
    index = filename.split('.')[0]
    image = Image.open(path)

    bar.set_postfix(**{'filename': filename, 'size': 'x'.join(map(str, image.size))})
    inputs = list(fill(image, input_shape))[0]
    inputs = np.expand_dims(inputs, axis=0)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.to(torch.float32)

    width, height = image.size
    with torch.no_grad():
        outputs = net(inputs)
        outputs = outputs[0]
        np.save(os.path.join('output', 'arrays', f'{index}.npy'), outputs.cpu().numpy())

        outputs = functional.softmax(outputs.permute(1, 2, 0), dim=-1).cpu().numpy()
        outputs = outputs[:height, :width]
        if args.detail:
            for clazz in range(21):
                frame = outputs[:, :, clazz]
                frame[frame >= 0.85] = clazz
                frame[frame < 0.85] = 0
                frame = frame.astype(np.uint8)
                mask = np.reshape(colors[frame.flatten()], [height, width, -1])
                mask = Image.fromarray(np.uint8(mask))

                folder = os.path.join('output', 'masks', index)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                mask.save(os.path.join(folder, f'{clazz}_{classes[clazz]}.png'))
            continue
        else:
            outputs = outputs.argmax(axis=-1)
    mask = np.reshape(colors[outputs.flatten()], [height, width, -1])
    mask = Image.fromarray(np.uint8(mask))
    mask.save(os.path.join('output', 'masks', filename))

    combined = Image.blend(image, mask, 0.7)
    combined.save(os.path.join('output', 'images', filename))
