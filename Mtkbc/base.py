from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np
import time
import os
from pathlib import Path
from loggings import logger
import pickle
from typing import Dict, Tuple, List
import torch
from utils import make_head_and_tail_dicts, turn_head_and_tail_dicts_into_arr


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    @abstractmethod
    def set_entity_weights(self, weights):
        pass

    @abstractmethod
    def get_entity_weights(self):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:  # 预测实体的个数
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    try:
                        q = self.get_queries(these_queries)
                        scores = q @ rhs
                    except:
                        q, srt = self.get_queries(these_queries)
                        scores = q @ rhs + srt

                    targets = self.score(these_queries)

                    # print(scores.size(), targets.size())  # [500,7129] [500,1]
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item(), query[4].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class JointModel(nn.Module):

    def __init__(self, temporal_model: TKBCModel, static_model: TKBCModel):
        super(JointModel, self).__init__()
        self.temporal_model = temporal_model
        self.static_model = static_model

        self.static_model.set_entity_weights(self.temporal_model.get_entity_weights())

    def forward(self, x):
        ans1 = self.temporal_model(x)
        ans2 = self.static_model(x)
        return ans1, ans2


DATA_PATH = os.path.join(os.getcwd(), 'data')


class TemporalDataset(object):
    def __init__(self, name: str, joint=False):
        self.root = Path(DATA_PATH) / name
        self.joint = joint
        self.ent_dict_arr = None
        self.ent_dict = {}

        self.logger = logger

        self.data = {}
        name = name.split('/')[2]
        self.has_interval = True if name in ['yago11k', 'wikidata12k'] else False
        self.n_blocks_time = 0
        self.n_blocks_time_2 = 0
        self.n_time = 0
        self.n_entity = 0
        self.n_relation = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.training_facts = []
        self.validation_facts = []
        self.test_facts = []
        self.to_skip_final = {'lhs': {}, 'rhs': {}}
        self.stat_to_skip_final = {}

    def get_examples(self, split):
        if split == 'train':
            return np.array(self.training_triples)
        elif split == 'valid':
            return np.array(self.validation_triples)
        else:
            return np.array(self.test_triples)

    def get_train(self):
        # reciprocal learning
        # copy = np.copy(self.data['train'])
        # tmp = np.copy(copy[:, 0])
        # copy[:, 0] = copy[:, 2]
        # copy[:, 2] = tmp
        # copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        # train_data = np.vstack((self.data['train'], copy))
        if self.joint:
            self.ent_dict = make_head_and_tail_dicts(self.training_triples)
            self.ent_dict_arr = turn_head_and_tail_dicts_into_arr(self.ent_dict)

        return np.array(self.training_triples)

    def eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), interval=False,
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()  # [total,..]
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_relation // 2
            if interval:
                ranks = model.get_ranking_interval(q, self.to_skip_final[m], batch_size=500)
            else:
                ranks = model.get_ranking(q, self.to_skip_final[m], batch_size=500)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entity, self.n_relation, self.n_entity, self.n_time, self.n_blocks_time