import argparse
import colorsys
import itertools
import os
import time

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from src.net import DeepLabV3Plus
from src.util.general import fill, debug

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

image = Image.open(input_path)
print('Image: ', 'x'.join(map(str, image.size)), sep='')

input_shape = (512, 512)

inputs = list(fill(image, input_shape))[0]
inputs = np.expand_dims(inputs, axis=0)
inputs = torch.from_numpy(inputs)
inputs = inputs.to(torch.float32)
debug(inputs=inputs)

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
colors = [(x / 21, 1.0, 1.0) for x in range(21)]
colors = itertools.starmap(colorsys.hsv_to_rgb, colors)
colors = map(lambda x: tuple(map(lambda y: int(y * 255), x)), colors)
colors = list(colors)

_, filename = os.path.split(input_path)
width, height = image.size
with torch.no_grad():
    net.eval()
    outputs = net(inputs)
    debug(outputs=outputs)
    outputs = outputs[0]
    np.save(os.path.join('cached', f"raw_{filename.split('.')[0]}.npy"), outputs.cpu().numpy())
    print('矩阵已保存')

    outputs = functional.softmax(outputs.permute(1, 2, 0), dim=-1).cpu().numpy()
    outputs = outputs[:height, :width]
    outputs = outputs.argmax(axis=-1)
    debug(outputs=outputs)

np.savetxt(os.path.join('cached', f"array_{filename.split('.')[0]}.txt"), outputs, fmt='%d')
print('矩阵已保存')

mask = np.reshape(np.array(colors, np.uint8)[np.reshape(outputs, [-1])], [height, width, -1])
mask = Image.fromarray(np.uint8(mask))
print('Mask: ', 'x'.join(map(str, mask.size)), sep='')
mask.save(os.path.join('cached', f'mask_{filename}'))
print('Mask 已保存')

combined = Image.blend(image, mask, 0.7)
combined.save(os.path.join('cached', f'output_{filename}'))
print('叠加图已保存')
