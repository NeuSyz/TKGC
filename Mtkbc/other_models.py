from typing import Tuple

import torch
from torch import nn
from base import TKBCModel


class DisMult(TKBCModel):

    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3, no_time_emb=False
    ):
        super(DisMult, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])

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

        return torch.sum(
            lhs * rel * rhs,
            1, keepdim=True
        )

    def forward(self, x):
        """
        :param x:
        :return: pred[batch,num_ent] , factors(s,p,o) , time
        """

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        right = self.embeddings[0].weight
        return (
            lhs * rel @ right.t(),
            rel,
            None
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):

        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return lhs * rel

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        pass


class TCP(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TCP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank, sparse=True),
            nn.Embedding(sizes[2], rank, sparse=True),
            nn.Embedding(sizes[3], rank, sparse=True),
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[2]
        self.time = self.embeddings[3]

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3])

        return torch.sum(
            lhs * rel * time * rhs,
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3])

        return (
            lhs * rel * time @ self.rhs.weight.t(),
            (lhs, rel, rhs),
            self.time.weight[:-1] if self.no_time_emb else self.time.weight
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])
        time = self.time(queries[:, 3])

        return lhs * rel * time

    def set_entity_weights(self, weights):
        pass

    def get_entity_weights(self):
        return self.embeddings[0].weight, self.embeddings[2].weight


class CP(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank, sparse=True),
            nn.Embedding(sizes[2], rank, sparse=True),
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[2]

    @staticmethod
    def has_time():
        return False

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(
            lhs * rel * rhs,
            1, keepdim=True
        )

    def forward(self, x):

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        # rhs = self.rhs(x[:, 2])

        return (
            lhs * rel @ self.rhs.weight.t(),
            (rel),
            None
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])

        return lhs * rel

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights[0]
        self.embeddings[2].weight = weights[1]

    def get_entity_weights(self):
        pass


class TRESCAL(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TRESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
            nn.Embedding(sizes[3], rank, sparse=True),
        ])

        nn.init.xavier_uniform_(self.embeddings[0].weight)
        nn.init.xavier_uniform_(self.embeddings[1].weight)
        nn.init.xavier_uniform_(self.embeddings[2].weight)

        self.no_time_emb = no_time_emb
        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]
        self.time = self.embeddings[2]

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3]).unsqueeze(1)
        full_rel = rel * time
        return torch.sum(
            torch.bmm(lhs.unsqueeze(1), full_rel).squeeze() * rhs,
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3]).unsqueeze(1)
        full_rel = rel * time

        return (
            (torch.bmm(lhs.unsqueeze(1), full_rel)).squeeze() @ self.rhs.weight.t(),
            [(lhs, full_rel, rhs)],
            self.time.weight[:-1] if self.no_time_emb else self.time.weight
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1]).reshape(-1, self.rank, self.rank)
        time = self.time(queries[:, 3]).unsqueeze(1)
        full_rel = rel * time

        return torch.bmm(lhs.unsqueeze(1), full_rel).squeeze()

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        return self.embeddings[0].weight


class RESCAL(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
        ])

        nn.init.xavier_uniform_(self.embeddings[0].weight)
        nn.init.xavier_uniform_(self.embeddings[1].weight)

        self.no_time_emb = no_time_emb
        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])

        return (
            (torch.bmm(lhs.unsqueeze(1), rel)).squeeze() @ self.rhs.weight.t(),
            [(lhs, rel, rhs)],
            None
        )

    @staticmethod
    def has_time():
        return False

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        return torch.sum(
            torch.bmm(lhs.unsqueeze(1), rel).squeeze() * rhs,
            1, keepdim=True
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1]).reshape(-1, self.rank, self.rank)

        return torch.bmm(lhs.unsqueeze(1), rel).squeeze()

    def set_entity_weights(self, weights):
        self.embeddings[0].weight = weights

    def get_entity_weights(self):
        pass


class TDisMult(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-3
    ):
        super(TDisMult, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank, sparse=True),
            nn.Embedding(sizes[3], rank, sparse=True),
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]
        self.time = self.embeddings[2]

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3])

        return torch.sum(
            lhs * rel * time * rhs,
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        time = self.time(x[:, 3])

        return (
            lhs * rel * time @ self.rhs.weight.t(),
            (lhs, rel, rhs),
            self.time.weight[:-1] if self.no_time_emb else self.time.weight
        )

    def forward_over_time(self, x):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])
        time = self.time(queries[:, 3])

        return lhs * rel * time

    def set_entity_weights(self, weights):
        pass

    def get_entity_weights(self):
        return self.embeddings[0].weight
