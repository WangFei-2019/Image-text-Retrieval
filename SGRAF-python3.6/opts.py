"""Argument parser"""

import argparse
import random
import os

def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='/workspace/Image-Text_Retrieval/datasets',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='/workspace/Image-Text_Retrieval/SGRAF-python3.6/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='/workspace/Image-Text_Retrieval/SGRAF-python3.6/runs10/{}_{}/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='/workspace/Image-Text_Retrieval/SGRAF-python3.6/runs10/{}_{}/log',
                        help='Path to save Tensorboard log.')
    # ---------------------------------------------------------------#
    parser.add_argument('--seed', default=None, help='default: random')  # 2022/01/20 19:28 增加随机种子
    parser.add_argument('--cuda', default='0', help='default: random')  # 2022/01/20 增加GPU选取

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=30, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')

    opt = parser.parse_args()

    if opt.seed is None:  # 2022/01/20 19:28 增加随机种子为文件命名
        opt.seed = random.randint(0, 100000)
    opt.logger_name = opt.logger_name.format(opt.data_name, opt.module_name) + '_' + str(opt.seed)
    opt.model_name = opt.model_name.format(opt.data_name, opt.module_name) + '_' + str(opt.seed)
    opt.seed = int(opt.seed)

    opt.data_name = opt.data_name + '_precomp'

    print(opt)
    return opt
