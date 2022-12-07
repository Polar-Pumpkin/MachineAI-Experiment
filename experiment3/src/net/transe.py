from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import LongTensor


class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int,
                 margin: float, device: Union[torch.device, None] = None):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dimension = dimension
        self.margin = margin
        self.device = device
        self.L = 2

        self.entity_embedding = nn.Embedding(num_embeddings=num_entities, embedding_dim=dimension)
        self.relation_embedding = nn.Embedding(num_embeddings=num_relations, embedding_dim=dimension)
        self.dist_func = nn.PairwiseDistance(self.L)

        self.__normalize(self.entity_embedding)
        self.__normalize(self.relation_embedding)
        if device is not None:
            self.entity_embedding = self.entity_embedding.to(device=device)
            self.relation_embedding = self.relation_embedding.to(device=device)
            self.dist_func = self.dist_func.to(device=device)

    @staticmethod
    def __normalize(embedding: nn.Embedding):
        # embedding.weight (Tensor) - 形状为 (num_embeddings, embedding_dim) 的嵌入中可学习的权值
        nn.init.xavier_uniform_(embedding.weight.data)
        norm = embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        embedding.weight.data.copy_(torch.from_numpy(norm))

    def score(self, triple: Union[LongTensor, Tuple[int, int, int]], device: Union[torch.device, None] = None):
        if isinstance(triple, tuple):
            triple = torch.from_numpy(np.array([triple])).long()
        h, r, t = torch.chunk(triple, 3, 1)
        if device is not None:
            h, r, t = map(lambda x: x.to(device=device), [h, r, t])

        h = torch.squeeze(self.entity_embedding(h), dim=1)
        r = torch.squeeze(self.relation_embedding(r), dim=1)
        t = torch.squeeze(self.entity_embedding(t), dim=1)
        return self.dist_func(h + r, t)

    def forward(self, triple, corrupted_triple):
        size = triple.size()[0]
        pos = self.score(triple, self.device)
        neg = self.score(corrupted_triple, self.device)
        return torch.sum(functional.relu(pos - neg + self.margin)) / size
