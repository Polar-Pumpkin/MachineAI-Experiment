import colorsys
import itertools
import os
from typing import Tuple

import cv2
import numpy as np

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
colors = [(x / 20, 1.0, 1.0) for x in range(21)]
colors = itertools.starmap(colorsys.hsv_to_rgb, colors)
colors = map(lambda x: tuple(map(lambda y: int(y * 255), x)), colors)
colors = list(colors)
colors[0] = (0, 0, 0)


def bgr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return b, g, r


white = (255, 255, 255)
black = (0, 0, 0)

padding = 10
width, height = (60, 20)
canvas = np.zeros(((height * 21) + (padding * 2), 240, 3), dtype=np.uint8)
canvas.fill(255)
for clazz, name in enumerate(classes):
    x = padding
    y = clazz * height + padding

    color = colors[clazz]
    cv2.rectangle(canvas, (x, y), (x + width, y + height), bgr(*color), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), white)
    cv2.putText(canvas, name, (x + width + padding, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, black)
cv2.imshow('Legends', canvas)
cv2.imwrite(os.path.join('cached', 'legends.png'), canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()
