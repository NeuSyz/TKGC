import argparse
from typing import Dict
from loggings import logger
import torch
import numpy as np
from torch import optim
from Dataset import KnowledgeGraph, MultiKnowledgeGraph
from Dataset_YG import KnowledgeGraphYG
from optimizers import TKBCOptimizer, JointOptimizer, IKBCOptimizer, MTKBCOptimizer, MJointOptimizer
from base import JointModel
from models import ComplEx, TComplEx, TNTComplEx, MTComplEx, DEComplEx, MTimeplex
from regularizers import N3, Lambda3, DURA_RESCAL_W, DURA_RESCAL
from other_models import TRESCAL, TCP, TDisMult, DisMult, RESCAL, CP
from tqdm import tqdm
from utils import setup_seed


setup_seed(124)

MODELS = {'ComplEx': TComplEx, 'TNTComplEx': TNTComplEx, 'RESCAL': TRESCAL, 'CP': TCP, 'DisMult': TDisMult,
          'MTComplEx': MTComplEx, 'DEComplEx': DEComplEx, 'Timeplex':MTimeplex}


S_MODELS = {'ComplEx': ComplEx, 'RESCAL': RESCAL, 'CP': CP, 'DisMult': DisMult, 'TNTComplEx': ComplEx,
            'MTComplEx': ComplEx}

# regularizer selection
REGULARIZERS = {'Norm3': N3, 'DURA_RESCAL_W': DURA_RESCAL_W, 'DURA_RESCAL': DURA_RESCAL}


def get_parser():
    parser = argparse.ArgumentParser(description="Temporal KGC")
    parser.add_argument(
        '--dataset', type=str, default='yago11k', choices=['icews14', 'icews05-15', 'gdelt', 'yago15k', 'wikidata12k', 'yago11k'],
        help="Dataset name"
    )
    models = [
        'ComplEx', 'TNTComplEx',  'RESCAL', 'CP', 'DisMult'
    ]
    parser.add_argument(
        '--model', default='TNTComplEx',
        help="Model in {}".format(models)
    )
    parser.add_argument(
        '--joint', action='store_true',
        help="use joint model"
    )
    parser.add_argument(
        '--typed', action='store_true',
        help="use typed joint model"
    )
    parser.add_argument(
        '--loss_ratio', default=0.6, type=float,
        help="joint model loss ratio"
    )
    parser.add_argument(
        '--max_epochs', default=500, type=int, choices=[100, 200, 500],
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid_freq', default=5, type=int, choices=[5, 10],
        help="Number of epochs between each valid."
    )
    parser.add_argument(
        '--early_stop', default=5, type=int, choices=[5, 10],
        help="early stopping."
    )
    parser.add_argument(
        '--rank', default=200, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--batch_size', default=2000, type=int, choices=[1000, 500],
        help="Batch size."
    )
    parser.add_argument(
        '--lr', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--init', default=1e-2, type=float, choices=[1e-3, 1e-2, 5e-3, 5e-2],
        help="Initial scale"
    )
    parser.add_argument(
        '--regularizer', default='Norm3', type=str,
        choices=['Norm3', 'DURA_RESCAL', 'DURA_RESCAL_W'],
    )
    parser.add_argument(
        '--emb_reg', default=1e-2, type=float,
        help="Embedding regularizer strength"
    )
    parser.add_argument(
        '--time_reg', default=1e-2, type=float,
        help="Timestamp regularizer strength"
    )
    parser.add_argument(
        '--thre',
        default=100, type=int,
        help='the mini threshold of time classes in yago11k and wikidata12k')

    parser.add_argument(
        '--alpha',
        default=1.0, type=float,
        help='Muti-time hyper')

    parser.add_argument(
        '--block_size',
        default=5, type=int,
        help='time block size')
    parser.add_argument(
        '--m', action='store_true',
        help="use multi-time embedding"
    )
    parser.add_argument(
        '--no_time_emb', action='store_true',
        help="Use a specific embedding for non temporal relations")
    return parser


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    hits = h.cpu().numpy().tolist()
    hits1 = hits[0]
    hits3 = hits[1]
    hits10 = hits[2]
    return {'MRR': m, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10}


def print_info(args):
    logger.info("Start Training ... \n")
    logger.info("--------Parameter Config-----------")
    logger.info("Dataset : {}".format(args.dataset))
    logger.info("Model : {}".format(args.model))
    logger.info("Rank : {}".format(args.rank))
    logger.info("learning rate : {}".format(args.lr))
    logger.info("emb_reg : {}".format(args.emb_reg))
    logger.info("time_reg : {}".format(args.time_reg))
    logger.info("alpha : {}".format(args.alpha))
    logger.info("block_size : {}".format(args.block_size))
    if args.joint:
        logger.info("****** Using Joint Model*******")
        logger.info("loss_ratio : {}".format(args.loss_ratio))


def train():
    parser = get_parser()
    args = parser.parse_args()
    print_info(args)

    """
    Data Loading
    """
    datadir = './data/' + args.dataset
    if args.dataset == 'yago11k' or args.dataset == 'wikidata12k':
        dataset = KnowledgeGraphYG(data_dir=datadir, count=args.thre)
    elif args.dataset == 'icews14' or args.dataset == 'icews05-15':
        if args.m:
            dataset = MultiKnowledgeGraph(data_dir=datadir, block_size=args.block_size)
        else:
            dataset = KnowledgeGraph(data_dir=datadir, joint=args.joint)

    elif args.dataset == 'yago15k':
        if args.m:
            dataset = MultiKnowledgeGraph(data_dir=datadir, block_size=args.block_size, no_t_emb=args.no_time_emb)
        else:
            dataset = KnowledgeGraph(data_dir=datadir, no_t_emb=args.no_time_emb, joint=args.joint)

    sizes = dataset.get_shape()

    if not args.joint:
        # model = S_MODELS[args.model]
        model = MODELS[args.model]
        model = model(
            sizes, args.rank,
            no_time_emb=args.no_time_emb,
            init_size=args.init,
            interval=dataset.has_interval,
            alpha=args.alpha,
        ).cuda()
    else:
        temp_model = MODELS[args.model]
        temp_model = temp_model(
            sizes,
            args.rank,
            no_time_emb=args.no_time_emb,
            init_size=args.init,
            interval=dataset.has_interval,
            alpha=args.alpha,
        )

        stat_model = S_MODELS[args.model]
        stat_model = stat_model(
            sizes,
            args.rank,
            init_size=args.init,
            interval=dataset.has_interval
        )

        model = JointModel(
            temporal_model=temp_model,
            static_model=stat_model
        ).cuda()

    opt = optim.Adagrad(model.parameters(), lr=args.lr)

    regularizer = REGULARIZERS[args.regularizer]
    emb_reg = regularizer(args.emb_reg)

    # timestamp reg
    time_reg = Lambda3(args.time_reg)

    # examples [total*2, 4]
    examples = dataset.get_train()

    early_stopping = args.early_stop
    best_mrr = -1
    cur_mrr = -1
    best_valid = {}
    best_test = {}
    with tqdm(total=args.max_epochs) as bar:

        for epoch in range(args.max_epochs):
            # 训练
            model.train()
            if args.joint:
                if args.m:
                    optimizer = MJointOptimizer(
                        model, emb_reg, time_reg, opt, dataset,
                        batch_size=args.batch_size, loss_ratio=args.loss_ratio,
                        type_ent=args.typed
                    )
                else:
                    optimizer = JointOptimizer(
                        model, emb_reg, time_reg, opt, dataset,
                        batch_size=args.batch_size, loss_ratio=args.loss_ratio,
                        type_ent=args.typed
                    )
                loss = optimizer.epoch(examples)
            else:
                if dataset.has_interval:
                    optimizer = IKBCOptimizer(
                        model, emb_reg, time_reg, opt, dataset,
                        batch_size=args.batch_size
                    )
                    # 开始训练
                    loss = optimizer.epoch(examples)
                else:
                    if args.m:
                        optimizer = MTKBCOptimizer(
                            model, emb_reg, time_reg, opt,
                            batch_size=args.batch_size
                    )
                    else:
                        optimizer = TKBCOptimizer(
                            model, emb_reg, time_reg, opt,
                            batch_size=args.batch_size
                        )
                    loss = optimizer.epoch(examples)

            # eval
            if (epoch + 1) % args.valid_freq == 0 and (epoch+1) >= 5:
                if dataset.has_interval:
                    if not args.joint:
                        valid, test = [
                            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, interval=True))
                            for split in ['valid', 'test']
                        ]
                    else:  # joint
                        valid, test = [
                            avg_both(*dataset.eval(model.temporal_model, split, -1 if split != 'train' else 50000, interval=True))
                            for split in ['valid', 'test']
                        ]
                else:
                    if not args.joint:
                        valid, test = [
                            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, interval=False))
                            for split in ['valid', 'test']
                        ]
                    else:  # joint
                        valid, test = [
                            avg_both(*dataset.eval(model.temporal_model, split, -1 if split != 'train' else 50000, interval=False))
                            for split in ['valid', 'test']
                        ]

                cur_mrr = round(test['MRR'], 3)
                if best_mrr < cur_mrr:
                    best_mrr = cur_mrr
                    best_valid = valid
                    best_test = test
                    early_stopping = args.early_stop
                else:
                    early_stopping -= 1
                if early_stopping == 0:
                    break
                # print("valid (MRR, hits@1, hits@3, hits@10): {:.3f} {:.3f} {:.3f} {:.3f}".format(
                #     valid['MRR'], valid['hits@1'], valid['hits@3'], valid['hits@10']))

                # if epoch + 1 >= 100:
                #     embeddings = model.temporal_model.get_entity_weights()
                #     embeddings = embeddings.detach().cpu().numpy()
                #     np.save('./{}.npy'.format(epoch), embeddings)

            bar.update(1)

            bar.set_postfix(
                loss=f'{loss.item():.4f}',
                best_mrr=f'{best_mrr:.3f}',
                mrr=f'{cur_mrr:.3f}',
            )
    print("valid (MRR, hits@1, hits@3, hits@10): {:.3f} {:.3f} {:.3f} {:.3f}".format(
        best_valid['MRR'], best_valid['hits@1'], best_valid['hits@3'], best_valid['hits@10']))
    print("test (MRR, hits@1, hits@3, hits@10): {:.3f} {:.3f} {:.3f} {:.3f}".format(
        best_test['MRR'], best_test['hits@1'], best_test['hits@3'], best_test['hits@10']))
    # torch.save(model.state_dict(), './experiment_res/model_param.pkl')


if __name__ == '__main__':
    train()
