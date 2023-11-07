# encoding: utf-8
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from base import TemporalDataset
from utils import make_head_and_tail_dicts, turn_head_and_tail_dicts_into_arr


class KnowledgeGraph(TemporalDataset):
    def __init__(self, data_dir, joint=False, no_t_emb=False, block_size=5):
        super(KnowledgeGraph, self).__init__(data_dir, joint)
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}

        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0
        self.n_time = 0
        self.no_time_emb = no_t_emb  # if True use YAGO15K
        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        self.load_filters()
        '''construct pools after loading'''
        # self.training_triple_pool = set(self.training_triples)
        # self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        time_dict_file = 'time2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        self.logger.info('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        self.n_relation *= 2
        self.logger.info('#relation: {}'.format(self.n_relation//2))
        print('-----Loading time dict-----')
        time_df = pd.read_table(os.path.join(self.data_dir, time_dict_file), header=None)
        self.time_dict = dict(zip(time_df[0], time_df[1]))
        self.n_time = len(self.time_dict)+1 if self.no_time_emb else len(self.time_dict)
        self.times = list(self.time_dict.values())
        self.logger.info('#timestamps: {}'.format(self.n_time))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.training_triples.append([self.entity_dict[triple[0]],self.relation_dict[triple[1]],self.entity_dict[triple[2]],timestamp, 0])
            self.training_facts.append([self.entity_dict[triple[0]],self.relation_dict[triple[1]],self.entity_dict[triple[2]],timestamp,0])
            # inverse relation
            self.training_triples.append([self.entity_dict[triple[2]],self.relation_dict[triple[1]]+self.n_relation//2,self.entity_dict[triple[0]],timestamp,0])

        self.n_training_triple = len(self.training_triples)
        self.logger.info('#training triple: {}'.format(self.n_training_triple//2))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.validation_triples.append([self.entity_dict[triple[0]],self.relation_dict[triple[1]],self.entity_dict[triple[2]], timestamp, 0])
            self.validation_facts.append([self.entity_dict[triple[0]],self.relation_dict[triple[1]],self.entity_dict[triple[2]], timestamp,0])

        self.n_validation_triple = len(self.validation_triples)
        self.logger.info('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.test_triples.append(
                    [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp, 0])
            self.test_facts.append([self.entity_dict[triple[0]],self.relation_dict[triple[1]],self.entity_dict[triple[2]], timestamp,0])

        self.n_test_triple = len(self.test_triples)
        self.logger.info('#test triple: {}'.format(self.n_test_triple))

    def load_filters(self):
        self.logger.info("creating filtering lists")
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        stat_to_skip = defaultdict(set)
        facts_pool = [self.training_facts,self.validation_facts,self.test_facts]
        for facts in facts_pool:
            for fact in facts:
                to_skip['lhs'][(fact[2], int(fact[1]+self.n_relation//2), fact[3], fact[4])].add(fact[0])  # left prediction
                # to_skip['lhs'][(fact[2], int(fact[1] + self.n_relation // 2), int(fact[3]+self.n_time//2), fact[4])].add(
                #     fact[0])  # left prediction
                to_skip['rhs'][(fact[0], fact[1], fact[3], fact[4])].add(fact[2])  # right prediction
                stat_to_skip[(fact[2], int(fact[1]+self.n_relation//2))].add(fact[0])
                stat_to_skip[(fact[0], fact[1])].add(fact[2])
                
        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip_final[kk][k] = sorted(list(v))
        for k, v in stat_to_skip.items():
            self.stat_to_skip_final[k] = sorted(list(v))
        self.logger.info("data preprocess completed")


class MultiKnowledgeGraph(TemporalDataset):
    def __init__(self, data_dir, joint=False, no_t_emb=False, block_size=5):
        super(MultiKnowledgeGraph, self).__init__(data_dir, joint)
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}

        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0
        self.n_time = 0
        self.block_size = block_size
        self.no_time_emb = no_t_emb  # if YAGO15K is True
        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        self.load_filters()
        '''construct pools after loading'''
        # self.training_triple_pool = set(self.training_triples)
        # self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        time_dict_file = 'time2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        self.logger.info('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        self.n_relation *= 2
        self.logger.info('#relation: {}'.format(self.n_relation // 2))
        print('-----Loading time dict-----')
        time_df = pd.read_table(os.path.join(self.data_dir, time_dict_file), header=None)
        self.time_dict = dict(zip(time_df[0], time_df[1]))
        self.n_time = len(self.time_dict) + 1 if self.no_time_emb else len(self.time_dict)
        self.times = list(self.time_dict.values())

        self.n_blocks_time = self.get_num_blocks(self.n_time)

        self.logger.info('#timestamps: {}'.format(self.n_time))

    def get_num_blocks(self, num_timestamps):
        num_blocks = num_timestamps // self.block_size + 1
        return num_blocks

    def TransMutiTime(self, time):
        """时间戳转为多时间粒度"""
        block_time = time // self.block_size
        return block_time

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.training_triples.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp,
                 self.TransMutiTime(timestamp)])
            self.training_facts.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp,
                 self.TransMutiTime(timestamp)])
            # inverse relation
            self.training_triples.append(
                [self.entity_dict[triple[2]], self.relation_dict[triple[1]] + self.n_relation // 2,
                 self.entity_dict[triple[0]], timestamp, self.TransMutiTime(timestamp)])

        self.n_training_triple = len(self.training_triples)
        self.logger.info('#training triple: {}'.format(self.n_training_triple // 2))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.validation_triples.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp, self.TransMutiTime(timestamp)])
            self.validation_facts.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp, self.TransMutiTime(timestamp)])

        self.n_validation_triple = len(self.validation_triples)
        self.logger.info('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            timestamp = self.time_dict[triple[3]] if triple[3] != 0 else len(self.time_dict)
            self.test_triples.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp, self.TransMutiTime(timestamp)])
            self.test_facts.append(
                [self.entity_dict[triple[0]], self.relation_dict[triple[1]], self.entity_dict[triple[2]], timestamp, self.TransMutiTime(timestamp)])

        self.n_test_triple = len(self.test_triples)
        self.logger.info('#test triple: {}'.format(self.n_test_triple))

    def load_filters(self):
        self.logger.info("creating filtering lists")
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        stat_to_skip = defaultdict(set)
        facts_pool = [self.training_facts, self.validation_facts, self.test_facts]
        for facts in facts_pool:
            for fact in facts:
                to_skip['lhs'][(fact[2], int(fact[1] + self.n_relation // 2), fact[3], fact[4])].add(
                    fact[0])  # left prediction
                # to_skip['lhs'][(fact[2], int(fact[1] + self.n_relation // 2), int(fact[3]+self.n_time//2), fact[4])].add(
                #     fact[0])  # left prediction
                to_skip['rhs'][(fact[0], fact[1], fact[3], fact[4])].add(fact[2])  # right prediction
                stat_to_skip[(fact[2], int(fact[1] + self.n_relation // 2))].add(fact[0])
                stat_to_skip[(fact[0], fact[1])].add(fact[2])

        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip_final[kk][k] = sorted(list(v))
        for k, v in stat_to_skip.items():
            self.stat_to_skip_final[k] = sorted(list(v))
        self.logger.info("data preprocess completed")
