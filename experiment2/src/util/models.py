import os

import torch
import torch.utils.model_zoo as model_zoo


def pretrained(net: type, url: str, map_location: torch.serialization.MAP_LOCATION = None, **kwargs):
    root = os.path.join('', 'cached')
    if not os.path.exists(root):
        os.makedirs(root)
    filename = url.split('/')[-1]

    path = os.path.join(root, filename)
    if os.path.exists(path):
        state = torch.load(path, map_location=map_location)
    else:
        state = model_zoo.load_url(url, model_dir=root, map_location=map_location)

    net = net(**kwargs)
    net.load_state_dict(state, False)
    return net
