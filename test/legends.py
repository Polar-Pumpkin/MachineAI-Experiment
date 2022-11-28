import colorsys
import itertools
from typing import Tuple

import cv2
import numpy as np

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
colors = [(x / 22, 1.0, 1.0) for x in range(21)]
colors = itertools.starmap(colorsys.hsv_to_rgb, colors)
colors = map(lambda x: tuple(map(lambda y: int(y * 255), x)), colors)
colors = list(colors)


def bgr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return b, g, r


legend_padding = 10
legend_width, legend_height = (60, 20)
legends = np.zeros(((legend_height * 21) + (legend_padding * 2), 240, 3), dtype=np.uint8)
for clazz, name in enumerate(classes):
    x = legend_padding
    y = clazz * legend_height + legend_padding

    color = colors[clazz]
    cv2.rectangle(legends, (x, y), (x + legend_width, y + legend_height), bgr(*color), -1)
    cv2.rectangle(legends, (x, y), (x + legend_width, y + legend_height), bgr(*(0, 0, 0)))
    cv2.putText(legends, name, (x + legend_width + legend_padding, y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255))
cv2.imshow('Legends', legends)

cv2.waitKey(0)
cv2.destroyAllWindows()
