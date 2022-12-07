import copy
import random
from typing import Union

import numpy as np
import torch

from ..data.wn18 import WN18Dataset, WN18Definitions


def poll(size: int, dataset: WN18Dataset, definitions: WN18Definitions, device: Union[torch.device, None] = None):
    samples = random.sample(dataset, size)
    current = []
    corrupted = []
    for sample in samples:
        sample = list(sample)
        corrupted_sample = copy.deepcopy(sample)
        h_c, r_c, t_c = corrupted_sample
        pr = np.random.random(1)[0]
        p = dataset.relation_tph[r_c] / (dataset.relation_tph[r_c] + dataset.relation_hpt[r_c])

        # 这里对于 p 的说明: tph 表示每一个头结对应的平均尾节点数, hpt 表示每一个尾节点对应的平均头结点数
        # 当tph > hpt 时更倾向于替换头, 反之则更倾向于替换尾实体

        # 举例说明:
        # 在一个知识图谱中, 一共有 10 个实体和 n 个关系, 如果其中一个关系使两个头实体对应五个尾实体，
        # 那么这些头实体的平均 tph 为 2.5, 而这些尾实体的平均 hpt 只有 0.4
        # 则此时我们更倾向于替换头实体, 因为替换头实体才会有更高概率获得正假三元组
        # 如果替换头实体, 获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
        if pr < p:
            # Change the head entity
            corrupted_sample[0] = definitions.get_entity_index(random.sample(definitions.entities, 1)[0])
            while corrupted_sample[0] == sample[0]:
                corrupted_sample[0] = definitions.get_entity_index(random.sample(definitions.entities, 1)[0])
        else:
            # Change the tail entity
            corrupted_sample[2] = definitions.get_entity_index(random.sample(definitions.entities, 1)[0])
            while corrupted_sample[2] == sample[2]:
                corrupted_sample[2] = definitions.get_entity_index(random.sample(definitions.entities, 1)[0])
        current.append(sample)
        corrupted.append(corrupted_sample)
    _current: torch.Tensor = torch.from_numpy(np.array(current)).long()
    _corrupted: torch.Tensor = torch.from_numpy(np.array(corrupted)).long()

    if device is not None:
        _current = _current.to(device)
        _corrupted = _corrupted.to(device)
    return _current, _corrupted
