import os
import re
from collections.abc import Sequence
from typing import List, Tuple, Dict, Iterator

from torch.utils.data import IterableDataset


class WN18Definitions:
    def __init__(self, path: str):
        self.path: str = path
        self.entities: List[int] = []
        self.relations: List[str] = ['_hyponym', '_hypernym', '_member_holonym', '_derivationally_related_form',
                                     '_instance_hypernym', '_also_see', '_member_meronym', '_member_of_domain_topic',
                                     '_part_of', '_instance_hyponym', '_synset_domain_topic_of', '_has_part',
                                     '_member_of_domain_usage', '_member_of_domain_region', '_synset_domain_usage_of',
                                     '_synset_domain_region_of', '_verb_group', '_similar_to']
        self.definitions: Dict[int, Tuple[int, str, str, int, str]] = {}

        self.load(path)

    def load(self, path: str):
        regex = r"^__(?P<word>.+?)_(?P<POS>[A-Z]{2})_(?P<index>\d+)$"
        with open(path, 'r') as file:
            for line in file.readlines():
                entityId, definition, description = line.split('\t')
                matches = re.search(regex, definition)
                assert matches, f'无法解析的实体定义: {definition}'
                word, pos, index = map(lambda x: matches.group(x), ['word', 'POS', 'index'])

                entityId = int(entityId)
                self.entities.append(entityId)
                self.definitions[entityId] = (entityId, word, pos, int(index), description)
        print(f'从 {path} 中加载 {len(self)} 条实体定义')

    def __len__(self) -> int:
        return len(self.definitions)

    def get_entity_id(self, entity: int) -> int:
        return self.entities.index(entity)

    def get_relation_id(self, relation: str) -> int:
        return self.relations.index(relation)


class WN18Dataset(IterableDataset[Tuple[int, int, int]], Sequence):

    def __init__(self, path: str, flag: str, definitions: WN18Definitions):
        super(WN18Dataset, self).__init__()
        assert os.path.exists(path), f'WN18 数据集路径无效: {path}'
        assert flag in ['train', 'valid', 'test'], f'未知的 WN18 数据集分类: {flag}'

        filename = f'wordnet-mlj12-{flag}.txt'
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), f'未找到 WN18 数据集分类文件: {file_path}'

        self.definitions: WN18Definitions = definitions
        self.triples: List[Tuple[int, str, int]] = []
        self.mapped_triples: List[Tuple[int, int, int]] = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                h, r, t = line.split('\t')
                h = int(h)
                t = int(t)
                self.triples.append((h, r, t))

                _h = definitions.get_entity_id(h)
                _r = definitions.get_relation_id(r)
                _t = definitions.get_entity_id(t)
                self.mapped_triples.append((_h, _r, _t))

                self.relation_head: Dict[int, Dict[int, int]] = {}
                if _r in self.relation_head:
                    if _h in self.relation_head[_r]:
                        self.relation_head[_r][_h] += 1
                    else:
                        self.relation_head[_r][_h] = 1
                else:
                    self.relation_head[_r]: Dict[int, int] = {}
                    self.relation_head[_r][_h] = 1

                self.relation_tail: Dict[int, Dict[int, int]] = {}
                if _r in self.relation_tail:
                    if _t in self.relation_tail[_r]:
                        self.relation_tail[_r][_t] += 1
                    else:
                        self.relation_tail[_r][_t] = 1
                else:
                    self.relation_tail[_r]: Dict[int, int] = {}
                    self.relation_tail[_r][_t] = 1
        print(f'从 {file_path} 中加载 {len(self)} 条三元组定义')

        self.relation_tph: Dict[int, float] = {}
        for r in self.relation_head:
            values = self.relation_head[r]
            self.relation_tph[r] = sum(values.values()) / len(values)

        self.relation_hpt: Dict[int, float] = {}
        for r in self.relation_tail:
            values = self.relation_tail[r]
            self.relation_hpt[r] = sum(values.values()) / len(values)

    def __getitem__(self, index) -> Tuple[int, int, int]:
        return self.mapped_triples[index]

    def __len__(self) -> int:
        return len(self.triples)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        return self.mapped_triples.__iter__()
