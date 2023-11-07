import numpy as np
import random
from collections import defaultdict
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_type_labels(r_idx, ent_dict_arr, n_entities):

    batch_size = r_idx.shape[0]
    type_labels = np.zeros((batch_size, n_entities), dtype=np.float32)
    for i in range(batch_size):
        type_labels[i, ent_dict_arr[r_idx[i]]] = 1

    return torch.from_numpy(type_labels).cuda()  # [batch, n_ent]


def generate_dual_labels(e_idx, r_idx, to_skip, n_entities):

    batch_size = r_idx.shape[0]
    raw_labels = np.zeros((batch_size, n_entities), dtype=np.float32)
    for i in range(batch_size):
        raw_labels[i, to_skip[(e_idx[i], r_idx[i])]] = 1

    return torch.from_numpy(raw_labels).cuda()  # [batch, n_ent]


def make_head_and_tail_dicts(data):

    ent_dict = defaultdict(set)
    for row in data:
        # e1, r, e2, ts = row
        e1, r, e2, ts, te = row
        ent_dict[r].add(e2)

    return dict(ent_dict)


def turn_head_and_tail_dicts_into_arr(ent_dict):

    ent_dict_arr = {}

    for r in ent_dict:  # r should also be in tail_dict
        e_indeces = np.empty(len(ent_dict[r]), dtype=np.uint32)

        for i, idx in enumerate(ent_dict[r]):
            e_indeces[i] = np.uint32(idx)
        ent_dict_arr[r] = e_indeces

    return ent_dict_arr


def complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im):  # <s,r,o_conjugate> when dim(s)==dim(r)==dim(o)
    sro = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
    return sro.sum(dim=-1)


def complex_3way_fullsoftmax(s_re, s_im, r_re, r_im):
    tmp1 = (s_im * r_re + s_re * r_im)
    tmp2 = (s_re * r_re - s_im * r_im)
    return tmp1, tmp2


def complex_hadamard(a_re, a_im, b_re, b_im):
    result_re = a_re * b_re - a_im * b_im
    result_im = a_re * b_im + a_im * b_re

    return result_re, result_im


if __name__ == '__main__':
    a = torch.tensor([[1,2],[3,4]])
    b = torch.tensor([2, 3])
    print(a * b.unsqueeze(0))