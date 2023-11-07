# Copyright (c) Facebook, Inc. and its affiliates.

import tqdm  # 进度条
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from base import TKBCModel
from regularizers import Regularizer
from base import TemporalDataset
from utils import generate_type_labels, generate_dual_labels


# class KBCOptimizer(object):
#     def __init__(
#             self, model: TKBCModel,
#             emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
#             optimizer: optim.Optimizer, dataset: TemporalDataset, batch_size: int = 256,
#             verbose: bool = True
#     ):
#         self.model = model
#         self.emb_regularizer = emb_regularizer
#         self.temporal_regularizer = temporal_regularizer
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.dataset = dataset
#
#     def epoch(self):
#
#         loss = nn.CrossEntropyLoss(reduction='mean')
#         b_begin = 0
#         l = None
#         while b_begin < examples.shape[0]:
#             input_batch = actual_examples[
#                           b_begin:b_begin + self.batch_size
#                           ].cuda()
#             predictions, factors, time = self.model.forward(input_batch)
#             truth = input_batch[:, 2]
#
#             l_fit = loss(predictions, truth)
#             l_reg = self.emb_regularizer.forward(factors)
#             l_time = torch.zeros_like(l_reg)
#             if time is not None:
#                 l_time = self.temporal_regularizer.forward(time)
#             l = l_fit + l_reg + l_time
#
#             self.optimizer.zero_grad()
#             l.backward()
#             self.optimizer.step()
#             b_begin += self.batch_size
#
#         return l


class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples):
        examples = torch.from_numpy(examples.astype('int64'))
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # shuffle
        loss = nn.CrossEntropyLoss(reduction='mean')
        b_begin = 0
        l = None
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + self.batch_size].cuda()
            predictions, factors, time = self.model.forward(input_batch)
            truth = input_batch[:, 2]

            l_fit = loss(predictions, truth)
            l_reg = self.emb_regularizer.forward(factors)

            # l_time1 = torch.zeros_like(l_reg)
            # l_time2 = torch.zeros_like(l_reg)
            # if time is not None:
            #     l_time1 = self.temporal_regularizer.forward(time[0])
            #     l_time2 = self.temporal_regularizer.forward(time[1])
            #
            # l = l_fit + l_reg + l_time1 + l_time2

            l_time = torch.zeros_like(l_reg)
            if time is not None:
                l_time = self.temporal_regularizer.forward(time)
            l = l_fit + l_reg + l_time

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size

        return l


class MTKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples):
        examples = torch.from_numpy(examples.astype('int64'))
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # shuffle
        loss = nn.CrossEntropyLoss(reduction='mean')
        b_begin = 0
        l = None
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + self.batch_size].cuda()
            predictions, factors, time = self.model.forward(input_batch)
            truth = input_batch[:, 2]

            l_fit = loss(predictions, truth)
            l_reg = self.emb_regularizer.forward(factors)

            l_time1 = torch.zeros_like(l_reg)
            l_time2 = torch.zeros_like(l_reg)
            l_time3 = torch.zeros_like(l_reg)
            if time is not None:
                l_time1 = self.temporal_regularizer.forward(time[0])
                l_time2 = self.temporal_regularizer.forward(time[1])
                l_time3 = self.temporal_regularizer.forward(time[1])

            l = l_fit + l_reg + l_time1 + l_time2 + l_time3

            # l_time = torch.zeros_like(l_reg)
            # if time is not None:
            #     l_time = self.temporal_regularizer.forward(time)
            # l = l_fit + l_reg + l_time

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size

        return l


class IKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset,batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_relation = dataset.n_relation

    def epoch(self, examples):
        # examples = torch.from_numpy(examples.astype('int64'))
        actual_examples = examples[np.random.permutation(examples.shape[0]), :]

        # actual_examples = examples[torch.randperm(examples.shape[0]), :]  # shuffle
        loss = nn.CrossEntropyLoss(reduction='mean')
        b_begin = 0
        l = None
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + self.batch_size]
            actual_batch = input_batch
            # end_miss = np.where(input_batch[:, 4:5] < 0)[0]
            # start_miss = np.where(input_batch[:, 3:4] < 0)[0]
            # actual_batch_e = np.delete(input_batch, 3, 1)  # end
            # actual_batch = np.delete(input_batch, 4, 1)  # start
            actual_batch_e = np.copy(actual_batch)
            actual_batch_e[:, 1:2] += self.n_relation  # 结束时间引入新关系
            # print(self.n_relation)
            actual_batch_e[:, 3] = actual_batch_e[:, 4]
            # actual_batch_e[end_miss, :] = actual_batch[end_miss, :]
            # actual_batch[start_miss, :] = actual_batch_e[start_miss, :]

            actual_input = torch.from_numpy(actual_batch.astype('int64')).cuda()
            actual_input_e = torch.from_numpy(actual_batch_e.astype('int64')).cuda()

            predictions, factors, time = self.model.forward(actual_input)
            predictions_e, factors_e, time_e = self.model.forward(actual_input_e)

            truth = actual_input[:, 2]
            # print(predictions.size())
            # print(truth.size())
            l_fit = loss(predictions, truth)
            l_fit += loss(predictions_e, truth)
            l_reg = self.emb_regularizer.forward(factors)
            l_reg += self.emb_regularizer.forward(factors_e)

            # l_time1 = torch.zeros_like(l_reg)
            # l_time2 = torch.zeros_like(l_reg)
            # if time is not None:
            #     l_time1 = self.temporal_regularizer.forward(time[0])
            #     l_time2 = self.temporal_regularizer.forward(time[1])
            #
            # l = l_fit + l_reg + l_time1 + l_time2

            l_time = torch.zeros_like(l_reg)
            if time is not None:
                l_time = self.temporal_regularizer.forward(time) + self.temporal_regularizer.forward(time_e)
            l = l_fit + l_reg + l_time

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size
        return l


class JointOptimizer(object):
    def __init__(
            self, joint_model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset, batch_size: int = 256,
            verbose: bool = True, loss_ratio=0.3, type_ent=False,
    ):
        self.joint_model = joint_model

        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_ratio = loss_ratio
        self.type_ent = type_ent
        self.dataset = dataset

    def epoch(self, examples):
        examples = torch.from_numpy(examples.astype('int64'))
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # shuffle
        loss = nn.CrossEntropyLoss(reduction='mean')
        stat_loss = nn.BCEWithLogitsLoss(reduction='mean')

        b_begin = 0
        l = None
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin: b_begin + self.batch_size].cuda()
            truth = input_batch[:, 2]
            if self.type_ent:
                stat_truth = generate_type_labels(
                    input_batch[:, 1].cpu().numpy(), self.dataset.ent_dict_arr, self.dataset.n_entity)
            else:
                stat_truth = generate_dual_labels(
                    input_batch[:, 0].cpu().numpy(), input_batch[:, 1].cpu().numpy(), self.dataset.stat_to_skip_final, self.dataset.n_entity)

            ans1, ans2 = self.joint_model(input_batch)
            predictions, factors, time = ans1
            stat_predictions, stat_factors, _ = ans2

            l_fit = loss(predictions, truth) + self.loss_ratio * stat_loss(stat_predictions, stat_truth)

            l_reg = self.emb_regularizer.forward(factors) + self.loss_ratio * self.emb_regularizer.forward(stat_factors)
            l_time = torch.zeros_like(l_reg)
            if time is not None:
                l_time = self.temporal_regularizer.forward(time)
            l = l_fit + l_reg + l_time

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size

        return l


class MJointOptimizer(object):
    def __init__(
            self, joint_model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, dataset: TemporalDataset, batch_size: int = 256,
            verbose: bool = True, loss_ratio=0.3, type_ent=False,
    ):
        self.joint_model = joint_model

        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_ratio = loss_ratio
        self.type_ent = type_ent
        self.dataset = dataset

    def epoch(self, examples):
        examples = torch.from_numpy(examples.astype('int64'))
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # shuffle
        loss = nn.CrossEntropyLoss(reduction='mean')
        stat_loss = nn.BCEWithLogitsLoss(reduction='mean')

        b_begin = 0
        l = None
        while b_begin < examples.shape[0]:
            input_batch = actual_examples[b_begin: b_begin + self.batch_size].cuda()
            truth = input_batch[:, 2]
            if self.type_ent:
                stat_truth = generate_type_labels(
                    input_batch[:, 1].cpu().numpy(), self.dataset.ent_dict_arr, self.dataset.n_entity)
            else:
                stat_truth = generate_dual_labels(
                    input_batch[:, 0].cpu().numpy(), input_batch[:, 1].cpu().numpy(), self.dataset.stat_to_skip_final, self.dataset.n_entity)

            ans1, ans2 = self.joint_model(input_batch)
            predictions, factors, time = ans1
            stat_predictions, stat_factors, _ = ans2

            l_fit = loss(predictions, truth) + self.loss_ratio * stat_loss(stat_predictions, stat_truth)

            l_reg = self.emb_regularizer.forward(factors) + self.loss_ratio * self.emb_regularizer.forward(stat_factors)
            l_time1 = torch.zeros_like(l_reg)
            l_time2 = torch.zeros_like(l_reg)
            if time is not None:
                l_time1 = self.temporal_regularizer.forward(time[0])
                l_time2 = self.temporal_regularizer.forward(time[1])
            l = l_fit + l_reg + l_time1 + l_time2

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            b_begin += self.batch_size

        return l
