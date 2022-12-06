import argparse
import os

import numpy as np
import torch

from src.data.wn18 import WN18Definitions
from src.net.transe import TransE

parser = argparse.ArgumentParser()
parser.add_argument('weights')

args = parser.parse_args()
weight_path = args.weights
if not os.path.exists(weight_path):
    print(f'模型权重文件路径不存在: {weight_path}')
    exit()

threshold = 0.85
dataset_root = os.path.join('.', 'datasets')
dataset_path = os.path.join(dataset_root, 'WN18')
definitions = WN18Definitions(os.path.join(dataset_path, 'wordnet-mlj12-definitions.txt'))

dimension = 50
margin = 1.0
net = TransE(len(definitions.entities), len(definitions.relations), dimension, margin)
missing, unexpected = net.load_state_dict(torch.load(weight_path))
if len(unexpected) > 0:
    print(f'模型权重文件冗余, 多余 {len(unexpected)} 条权值')
if len(missing) > 0:
    print(f'模型权重文件不完整, 缺少 {len(missing)} 条权值')
    exit()
print('模型权重已加载完毕')

word = input('请输入单词: ')
entity_id = definitions.get_entity_from_word(word)
if entity_id < 0:
    print('未查询到此单词')
    exit()

print('可用的查询模式:', '0 - 查单词 (根据关系)', '1 - 查关系 (根据另一单词)', sep='\n')
mode = input('请选择查询模式: ')
if not mode.isdigit():
    print('无效的查询模式')
    exit()
mode = int(mode)

if mode == 0:
    relation = input('请输入关系: ')
    if relation.isdigit():
        relation = int(relation)
        if relation >= len(definitions.relations):
            print('未知的关系')
            exit()
        relation = definitions.relations[relation]
    else:
        if relation not in definitions.relations:
            print('未知的关系')
            exit()
    count = 0
    for target_id, ref, pos, _, description in definitions.definitions.values():
        score = net.score(torch.from_numpy(np.array([(
            definitions.get_entity_id(entity_id),
            definitions.get_relation_id(relation),
            definitions.get_entity_id(target_id)
        )])).long()).item()
        print(f'{ref} -> {score}')
        count += 1
        if count >= 10:
            exit()
    # TODO
elif mode == 1:
    other = input('请输入单词: ')
    other_id = definitions.get_entity_from_word(other)
    if other_id < 0:
        print('未查询到此单词')
        exit()

    count = 0
    for relation in definitions.relations:
        score = net.score(torch.from_numpy(np.array([(
            definitions.get_entity_id(entity_id),
            definitions.get_relation_id(relation),
            definitions.get_entity_id(other_id)
        )])).long()).item()
        print(f'{relation} -> {score}')
        count += 1
        if count >= 10:
            exit()
    # TODO
else:
    print(f'未知的查询模式: {mode}')
