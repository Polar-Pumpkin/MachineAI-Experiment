from typing import Tuple, Union, List

import cv2
import numpy as np
import torch
from PIL import Image


def duration(seconds: int) -> str:
    descriptions = []
    hours = seconds // 3600
    if hours >= 24:
        days = hours // 24
        if days > 0:
            descriptions.append(f'{days} 天')
            hours %= 24
    if hours > 0:
        descriptions.append(f'{hours} 小时')
        seconds %= 3600
    minutes = seconds // 60
    if minutes > 0:
        descriptions.append(f'{minutes} 分钟')
        seconds %= 60
    descriptions.append(f'{seconds} 秒')
    return ' '.join(descriptions)


def fill(images: Union[Image.Image, List[Image.Image]], input_shape: Tuple[int, int],
         transforms: Union[bool, List[bool]] = True):
    width, height = input_shape

    def _fill(image: Image.Image, transform: bool):
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
        if transform:
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.0
        return image

    if isinstance(images, Image.Image):
        images = [images]
    size = len(images)

    if isinstance(transforms, bool):
        transforms = [transforms] * size
    return map(_fill, images, transforms)


def debug(**kwargs):
    for name, value in kwargs.items():
        print(f'{name} ({type(value)})', end=': ')

        if isinstance(value, torch.Tensor):
            shape = ', '.join(map(str, value.size()))
            print(f'[{shape}, {value.dtype}]', end=', ')
            print(f'Requires grad={value.requires_grad}', end=', ')
            print(f'Grad function={value.grad_fn}', end='')

        if isinstance(value, np.ndarray):
            shape = ', '.join(map(str, value.shape))
            print(f'[{shape}, {value.dtype}]', end='')

        print('\n', end='')
