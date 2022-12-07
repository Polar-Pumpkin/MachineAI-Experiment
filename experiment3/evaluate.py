import argparse
import os

import torch

from src.data.wn18 import WN18Definitions
from src.net.transe import TransE
from src.util.general import Colors

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


def select(inputs: str) -> int:
    references = definitions.get_entity_id_from_word(inputs)
    amount = len(references)
    if not amount > 0:
        print('未查询到此单词')
        exit()
    elif amount > 1:
        print(f'该单词存在 {amount} 个义项:')
        for ref_id in references:
            definitions.print(ref_id)
        num = input(f'请选择目标义项(1-{amount}): ')
        if not num.isdigit():
            print('无效的义项编号')
            exit()

        num = int(num) - 1
        if num >= amount:
            print('无效的义项编号')
            exit()
        return references[num]
    else:
        return references[0]


word = input(f'请输入单词: ')
entity_id = select(word)

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
        print(f'已选择关系: {Colors.BOLD}{relation}{Colors.END}')
    else:
        if relation not in definitions.relations:
            print('未知的关系')
            exit()

    count = 0
    for target_id, ref, pos, _, description in definitions.definitions.values():
        score = net.score(definitions.map(entity_id, relation, target_id)).item()
        print(f'{ref} -> {score}')
        count += 1
        if count >= 30:
            exit()
    # TODO
elif mode == 1:
    other = input('请输入单词: ')
    other_id = select(other)

    count = 0
    for relation in definitions.relations:
        score = net.score(definitions.map(entity_id, relation, other_id)).item()
        print(f'{relation} -> {score}')
        count += 1
        if count >= 30:
            exit()
    # TODO
else:
    print(f'未知的查询模式: {mode}')
