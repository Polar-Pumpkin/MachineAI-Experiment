import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..util import losses


class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int,
                 margin: float, norm: int, c: float,
                 device: torch.device = None):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dimension = dimension
        self.margin = margin
        self.norm = norm
        self.c = c
        self.device = device

        self.entity_embedding = nn.Embedding(num_embeddings=num_entities, embedding_dim=dimension)
        self.relation_embedding = nn.Embedding(num_embeddings=num_relations, embedding_dim=dimension)
        self.loss_func = nn.MarginRankingLoss(margin, reduction='mean')

        self.__normalize(self.entity_embedding)
        self.__normalize(self.relation_embedding)
        if device is not None:
            self.entity_embedding = self.entity_embedding.to(device=device)
            self.relation_embedding = self.relation_embedding.to(device=device)
            self.loss_func = self.loss_func.to(device=device)

    @staticmethod
    def __normalize(embedding: nn.Embedding):
        # embedding.weight (Tensor) - 形状为 (num_embeddings, embedding_dim) 的嵌入中可学习的权值
        nn.init.xavier_uniform_(embedding.weight.data)
        norm = embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        embedding.weight.data.copy_(torch.from_numpy(norm))

    def distance(self, h, r, t, test: bool = False):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作, 如果想要保持结果的维度不变, 设置参数 keepdim=True 即可
        # 如下面 sum 中 r_norm * h 结果是一个 1024x50 的矩阵(2 维张量) sum 在 dim 的结果就变成了 1024 的向量(1 维张量)
        # 如果想和 r_norm 对应元素两两相乘, 就需要 sum 的结果也是 2 维张量
        # 因此需要使用 keepdim=True 保证张量的维度不变
        # 另外对于 dim 等于几表示张量的第几个维度, 从 0 开始计数, 可以理解为张量的最开始的第几个左括号
        # 具体可以参考这个 https://www.cnblogs.com/flix/p/11262606.html
        head = self.entity_embedding(h)
        relation = self.relation_embedding(r)
        tail = self.entity_embedding(t)

        distance = head + relation - tail
        # dim=-1 表示的是维度的最后一维, 比如如果一个张量有 3 维, 那么 dim=2=-1, dim=0=-3

        score = torch.norm(distance, p=self.norm, dim=1)
        if test:
            score = score.detach().cpu().numpy()

        if self.device is not None:
            score = score.to(device=self.device)
        return score

    def forward(self, triple, corrupted_triple):
        h, r, t = torch.chunk(triple, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triple, 3, dim=1)

        h = torch.squeeze(h, dim=1)
        r = torch.squeeze(r, dim=1)
        t = torch.squeeze(t, dim=1)
        h_c = torch.squeeze(h_c, dim=1)
        r_c = torch.squeeze(r_c, dim=1)
        t_c = torch.squeeze(t_c, dim=1)
        if self.device is not None:
            h, r, t, h_c, r_c, t_c = map(lambda x: x.to(device=self.device), [h, r, t, h_c, r_c, t_c])

        # torch.nn.Embedding 类的 forward 只接受 LongTensor 类型的张量
        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        entity_embeded = self.entity_embedding(torch.cat([h, t, h_c, t_c]))
        relation_embeded = self.relation_embedding(torch.cat([r, r_c]))

        y = Variable(torch.Tensor([-1]))
        if self.device is not None:
            y = y.to(device=self.device)
        loss = self.loss_func(pos, neg, y)

        entity_scale_loss = losses.scale(entity_embeded, device=self.device)
        relation_scale_loss = losses.scale(relation_embeded, device=self.device)
        return loss + self.c * (entity_scale_loss / len(entity_embeded) + relation_scale_loss / len(relation_embeded))
