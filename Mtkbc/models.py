from typing import Tuple

import math
import torch
from torch import nn
from base import TKBCModel
from utils import complex_3way_simple, complex_3way_fullsoftmax


class ComplEx(TKBCModel):

    def __init__(
            self, sizes, rank: int,
            init_size: float = 1e-2, no_time_emb=False, interval=False
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        # embedding初始化
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        """评分函数计算"""
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        """
        :param x:
        :return: pred[batch,num_ent] , factors(s,p,o) , time
        """
        # x [batch,4]
        # lhs,rel,rhs [batch,2*r]
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        # rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):

        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight


class TComplEx(TKBCModel):
    def __init__(
            self, sizes, rank: int,
            no_time_emb=False, init_size: float = 1e-2, interval=False, alpha=1.0
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]*2 if interval else sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        # torch.nn.init.normal_(self.embeddings[0].weight.data, 0, 0.05)
        # torch.nn.init.normal_(self.embeddings[1].weight.data, 0, 0.05)
        # torch.nn.init.normal_(self.embeddings[2].weight.data, 0, 0.05)

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        # 将时间嵌入向量同样引入复数空间中，将时间嵌入与实体、关系进行点积得到依赖时间的实体和关系表征
        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        """
        :param x:
        :return: pred , factors：元组(s,p,o) , time
        """
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        # 考虑时间与关系进行点积，得到感知时间的关系表征
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight


class TNTComplEx(TKBCModel):
    """TNTComplEx"""
    def __init__(
            self, sizes, rank: int,
            no_time_emb=False, init_size: float = 1e-2, interval=False, alpha=1.0
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]*2 if interval else sizes[1], sizes[3], sizes[1]*2 if interval else sizes[1]]
        ])
        # self.embeddings[3] no-time relation embedding
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]  # 为关系加入了非时间嵌入

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        # math.pow(x,y) : x的y次方
        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight[:self.sizes[3]]
                # (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight[:self.sizes[3]//2],
                #  self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight[self.sizes[3]//2+1:])
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight


class DEComplEx(TKBCModel):
    def __init__(
            self, sizes, rank,
            no_time_emb=False, init_size: float = 1e-2, interval=False, alpha=1.0
    ):
        super(DEComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        self.propotion = 0.6

        self.E = torch.nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.T = torch.nn.Embedding(sizes[2], 2 * int(self.propotion * rank), sparse=True)
        self.R = torch.nn.Embedding(sizes[1], 2 * int(self.propotion * rank), sparse=True)
        self.NR = torch.nn.Embedding(sizes[1], 2 * int((1-self.propotion) * rank), sparse=True)

        self.sp = int(self.propotion * self.rank)
        self.ssp = int((1-self.propotion) * self.rank)

        # self.embeddings[3] no-time relation embedding
        self.E.weight.data *= init_size
        self.T.weight.data *= init_size
        self.R.weight.data *= init_size
        self.NR.weight.data *= init_size

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.NR(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.sp], rel[:, self.sp:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.sp], time[:, self.sp:]
        rnt = rel_no_time[:, :self.ssp], rel_no_time[:, self.ssp:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]

        full_rel = torch.cat((rrt[0],rnt[0]), dim=-1), torch.cat((rrt[1],rnt[1]), dim=-1)

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.NR(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.sp], rel[:, self.sp:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.sp], time[:, self.sp:]

        rnt = rel_no_time[:, :self.ssp], rel_no_time[:, self.ssp:]

        right = self.E.weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = torch.cat((rrt[0], rnt[0]), dim=-1), torch.cat((rrt[1], rnt[1]), dim=-1)

        # math.pow(x,y) : x的y次方
        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
                self.T.weight[:-1] if self.no_time_emb else self.T.weight[:self.sizes[3]]
                # (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight[:self.sizes[3]//2],
                #  self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight[self.sizes[3]//2+1:])
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.E.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.E(queries[:, 0])
        rel = self.R(queries[:, 1])
        rel_no_time = self.NR(queries[:, 1])
        time = self.T(queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.sp], rel[:, self.sp:]
        time = time[:, :self.sp], time[:, self.sp:]
        rnt = rel_no_time[:, :self.ssp], rel_no_time[:, self.ssp:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = torch.cat((rrt[0],rnt[0]), dim=-1), torch.cat((rrt[1],rnt[1]), dim=-1)

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

    def set_entity_weights(self, weights):
        self.E.weight = weights

    def get_entity_weights(self):
        return self.E.weight


class MTComplEx(TKBCModel):
    def __init__(
            self, sizes, rank: int,
            no_time_emb=False, init_size: float = 1e-2, interval=False, alpha=1.0
    ):
        super(MTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]*2 if interval else sizes[1], sizes[3], sizes[1]*2 if interval else sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

        self.m_time_embeddings = nn.Embedding(sizes[4], 2*rank, sparse=True)
        self.m_time_embeddings.weight.data *= init_size
        self.alpha = alpha

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        m_rel = self.embeddings[3](x[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]
        m_time = self.m_time_embeddings(x[:, 4])
        m_time = m_time[:, :self.rank], m_time[:, self.rank:]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rmt = m_rel[0] * m_time[0], m_rel[1] * m_time[0], m_rel[0] * m_time[1], m_rel[1] * m_time[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (rmt[0] - rmt[3]), (rt[1] + rt[2]) + self.alpha * (rmt[1] + rmt[2])

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        """
        :param x:
        :return: pred , factors：元组(s,p,o) , time
        """
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        m_time = self.m_time_embeddings(x[:, 4])
        m_time = m_time[:, :self.rank], m_time[:, self.rank:]
        m_rel = self.embeddings[3](x[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        mt = m_rel[0] * m_time[0], m_rel[1] * m_time[0], m_rel[0] * m_time[1], m_rel[1] * m_time[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (mt[0] - mt[3]), (rt[1] + rt[2]) + self.alpha * (mt[1] + mt[2])

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), (
            self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight,
            self.m_time_embeddings.weight
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        mtime = self.m_time_embeddings(queries[:, 4])
        m_rel = self.embeddings[3](queries[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        mtime = mtime[:, :self.rank], mtime[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rmt = m_rel[0] * mtime[0], m_rel[1] * mtime[0], m_rel[0] * mtime[1], m_rel[1] * mtime[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (rmt[0] - rmt[3]), (rt[1] + rt[2]) + self.alpha * (rmt[1] + rmt[2])

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight


class MTimeplex(TKBCModel):
    def __init__(
            self, sizes, rank: int,
            no_time_emb=False, init_size: float = 1e-2, interval=False, alpha=1.0
    ):
        super(MTimeplex, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]*2 if interval else sizes[1], sizes[3], sizes[1]*2 if interval else sizes[1]]
        ])

        self.m_time_embeddings = nn.Embedding(sizes[4], 2 * rank, sparse=True)
        self.m_time_embeddings.weight.data *= init_size
        self.Rs = nn.Embedding(sizes[1], 2 * rank, sparse=True)
        self.Ro = nn.Embedding(sizes[1], 2 * rank, sparse=True)
        self.Rs.weight.data *= init_size
        self.Ro.weight.data *= init_size

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.alpha = alpha

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        m_rel = self.embeddings[3](x[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]
        rs = self.Rs(x[:, 1])
        rs = rs[:, :self.rank], rs[:, self.rank:]
        ro = self.Ro(x[:, 1])
        ro = ro[:, :self.rank], ro[:, self.rank:]
        m_time = self.m_time_embeddings(x[:, 4])
        m_time = m_time[:, :self.rank], m_time[:, self.rank:]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rmt = m_rel[0] * m_time[0], m_rel[1] * m_time[0], m_rel[0] * m_time[1], m_rel[1] * m_time[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (rmt[0] - rmt[3]),\
                   (rt[1] + rt[2]) + self.alpha * (rmt[1] + rmt[2])

        sro = complex_3way_simple(lhs[0],lhs[1],full_rel[0],full_rel[1],rhs[0],rhs[1])
        srt = complex_3way_simple(lhs[0], lhs[1], rs[0], rs[1], time[0], time[1])
        ort = complex_3way_simple(time[0], time[1], ro[0], ro[1], rhs[0], rhs[1])
        sot = complex_3way_simple(lhs[0], lhs[1], time[0], time[1], rhs[0], rhs[1])
        score = sro + 1.0*srt + 1.0*ort + 1.0*sot
        return score.unsqueeze(1)

    def forward(self, x):
        """
        :param x:
        :return: pred , factors：元组(s,p,o) , time
        """
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        m_time = self.m_time_embeddings(x[:, 4])
        m_time = m_time[:, :self.rank], m_time[:, self.rank:]
        m_rel = self.embeddings[3](x[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]

        rs = self.Rs(x[:, 1])
        rs = rs[:, :self.rank], rs[:, self.rank:]
        ro = self.Ro(x[:, 1])
        ro = ro[:, :self.rank], ro[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        mt = m_rel[0] * m_time[0], m_rel[1] * m_time[0], m_rel[0] * m_time[1], m_rel[1] * m_time[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (mt[0] - mt[3]), (rt[1] + rt[2]) + self.alpha * (mt[1] + mt[2])

        sro = complex_3way_fullsoftmax(lhs[0], lhs[1], full_rel[0], full_rel[1])
        srt = complex_3way_simple(lhs[0], lhs[1], rs[0], rs[1], time[0], time[1]).unsqueeze(1)
        ort = complex_3way_fullsoftmax(time[0], time[1], ro[0], ro[1])
        sot = complex_3way_fullsoftmax(lhs[0], lhs[1], time[0], time[1])

        return (
                    sro[0] @ right[0].t() + sro[1] @ right[1].t() +
                    1.0*(ort[0] @ right[0].t() + ort[1] @ right[1].t()) +
                    1.0*(sot[0] @ right[0].t() + sot[1] @ right[1].t()) +
                    1.0*(srt[0] + srt[1])
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
                   torch.sqrt(ro[0] ** 2 + ro[1] ** 2),
                   torch.sqrt(rs[0] ** 2 + rs[1] ** 2)
               ), (
            self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight,
            self.m_time_embeddings.weight
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        mtime = self.m_time_embeddings(queries[:, 4])
        m_rel = self.embeddings[3](queries[:, 1])
        m_rel = m_rel[:, :self.rank], m_rel[:, self.rank:]

        rs = self.Rs(queries[:, 1])
        rs = rs[:, :self.rank], rs[:, self.rank:]
        ro = self.Ro(queries[:, 1])
        ro = ro[:, :self.rank], ro[:, self.rank:]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        mtime = mtime[:, :self.rank], mtime[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rmt = m_rel[0] * mtime[0], m_rel[1] * mtime[0], m_rel[0] * mtime[1], m_rel[1] * mtime[1]

        full_rel = (rt[0] - rt[3]) + self.alpha * (rmt[0] - rmt[3]), (rt[1] + rt[2]) + self.alpha * (rmt[1] + rmt[2])

        sro = complex_3way_fullsoftmax(lhs[0], lhs[1], full_rel[0], full_rel[1])
        srt = complex_3way_simple(lhs[0], lhs[1], rs[0], rs[1], time[0], time[1]).unsqueeze(1)
        ort = complex_3way_fullsoftmax(time[0], time[1], ro[0], ro[1])
        sot = complex_3way_fullsoftmax(lhs[0], lhs[1], time[0], time[1])
        quer = torch.cat([
            sro[0] + 1.0 * ort[0] + 1.0 * sot[0],
            sro[1] + 1.0 * ort[1] + 1.0 * sot[1]
        ], 1)
        # print(srt.size())
        return quer, 5.0*srt

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight