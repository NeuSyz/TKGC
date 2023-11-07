# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


# Norm-3正则化器
class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


# 时间嵌入平滑
class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        # 相邻的时间戳嵌入距离更近
        ddiff = factor[1:] - factor[:-1]  # [time_num-1, emb]
        rank = int(ddiff.shape[1] / 2)
        # 也使用了N3正则化
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class DURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 2 + t ** 2)
            norm += torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)

        return self.weight * norm / h.shape[0]


class DURA_RESCAL_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += 2.0 * torch.sum(h ** 2 + t ** 2)
            norm += 0.5 * torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)

        return self.weight * norm / h.shape[0]


# class DURA(Regularizer):
#     def __init__(self, weight: float):
#         super(DURA, self).__init__()
#         self.weight = weight
#
#     def forward(self, factors):
#         norm = 0
#
#         for factor in factors:
#             h, r, t = factor
#
#             norm += torch.sum(t**2 + h**2)
#             norm += torch.sum(h**2 * r**2 + t**2 * r**2)
#
#         return self.weight * norm / h.shape[0]
#
#
# class DURA_W(Regularizer):
#     def __init__(self, weight: float):
#         super(DURA_W, self).__init__()
#         self.weight = weight
#
#     def forward(self, factors):
#         norm = 0
#         for factor in factors:
#             h, r, t = factor
#
#             norm += 0.5 * torch.sum(t**2 + h**2)
#             norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)
#
#         return self.weight * norm / h.shape[0]