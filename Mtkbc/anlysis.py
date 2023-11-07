from datasets import TemporalDataset
import re
from collections import defaultdict
from Dataset import KnowledgeGraph
from Dataset_YG import KnowledgeGraphYG

def train():
    datadir = './data/icews14'
    dataset = KnowledgeGraph(data_dir=datadir)
    dataset.training_facts.extend(dataset.validation_facts)
    dataset.training_facts.extend(dataset.test_facts)
    total_tris = dataset.training_facts
    rel_dic = dataset.relation_dict
    id2rel = {}
    for k,v in rel_dic.items():
        id2rel[v] = k
    rel_time = defaultdict(set)
    for lhs, rel, rhs, t, _ in total_tris:
        rel_time[rel].add(t)
    count = 0
    for k,v in rel_time.items():
        # print("{}-----{}".format(k,v))
        if len(v)<=3:
            print("{}-----{}".format(k, v))
            count += 1
    print(count)
    # if args.dataset == 'yago11k' or args.dataset == 'wikidata12k':
    #     dataset = KnowledgeGraphYG(data_dir=datadir, count=args.thre)
    #     total_times = dataset.n_time
    # elif args.dataset == 'icews14' or args.dataset == 'icews05-15':
    #     dataset = KnowledgeGraph(data_dir=datadir)
    # elif args.dataset == 'yago15k':
    #     dataset = KnowledgeGraph(data_dir=datadir, no_t_emb=args.no_time_emb)

    # dataset = TemporalDataset('ICEWS14')
    #
    # sizes = dataset.get_shape()
    # train_data = dataset.data['train']
    # rel_his = defaultdict(list)
    # rel2id =  {}
    # with open('./data/ICEWS14/rel_id', 'r') as f:
    #     for line in f.readlines():
    #         rel, index = line.split('\t')
    #         rel2id[index] = rel
    #
    # for tri in dataset.data['train']:
    #     h,r,t,tt = tri
    #     rel_his[r].append(tt)
    # new_rel = defaultdict(list)
    # for k,v in rel_his.items():
    #     new_rel[k] = sorted(v)
    #         # set(sorted(v))
    #
    # num_rel_5 = 0
    # num_rel_10 = 0
    # num_rel_15 = 0
    # num_rel_20 = 0
    # for k,v in new_rel.items():
    #     if len(v)<=10:
    #         num_rel_5+=1
    #     elif len(v)<=20:
    #         num_rel_10 += 1
    #     elif len(v)<=50:
    #         num_rel_15 += 1
    #     else:
    #         num_rel_20 +=1
    # print(num_rel_5)
    # print(num_rel_10)
    # print(num_rel_15)
    # print(num_rel_20)

if __name__ == '__main__':
    # text = "####-##-##"
    # begin = re.search(r'(-*\d+)-([#|\d]+)-([#|\d]+)', text)
    # print(begin.group(1))
    train()


