import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from src.net import DeepLabV3Plus
from src.util.general import fill

parser = argparse.ArgumentParser()

parser.add_argument('input', help='输入图像的路径')
parser.add_argument('-m', '--model', help='模型的路径')

args = parser.parse_args()

input_path = args.input
if not os.path.exists(input_path):
    print(f'无效的输入图像路径: {input_path}')
    exit()

if args.model is not None:
    model_path = args.model
else:
    latest = None
    for name, path in [(name, os.path.join('runs', name)) for name in os.listdir('runs')]:
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'best.pth')):
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

print(f'输入图像: {input_path}')
print(f'模型路径: {model_path}')

net = DeepLabV3Plus(21, pretrained=False)
net.load_state_dict(torch.load(model_path), False)

net.eval()
image = Image.open(input_path)
inputs = fill(image, (512, 512))[0]
inputs = np.expand_dims(inputs, axis=0)
inputs = torch.from_numpy(inputs)
print(type(inputs))
print(inputs.shape)

outputs = net(inputs)
print(type(outputs))
print(outputs.shape)
