import os
import time
import shutil

import torch
import numpy

import data
from model import SAEM
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from torch.autograd import Variable
from scipy.spatial.distance import cdist

import logging
from datetime import datetime
import tensorboard_logger as tb_logger

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/workspace/Image-Text_Retrieval/datasets',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f30k}_precomp')
    # parser.add_argument('--margin', default=0.2, type=float,
    #                     help='Rank loss margin.')
    # parser.add_argument('--num_epochs', default=30, type=int,
    #                     help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    # parser.add_argument('--word_dim', default=300, type=int,
    #                     help='Dimensionality of the word embedding.')
    # parser.add_argument('--embed_size', default=1024, type=int,
    #                     help='Dimensionality of the joint embedding.')
    # parser.add_argument('--grad_clip', default=2., type=float,
    #                     help='Gradient clipping threshold.')
    # parser.add_argument('--learning_rate', default=.0001, type=float,
    #                     help='Initial learning rate.')
    # parser.add_argument('--lr_update', default=10, type=int,
    #                     help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    # parser.add_argument('--log_step', default=10, type=int,
    #                     help='Number of steps to print and record the log.')
    # parser.add_argument('--val_step', default=1000, type=int,
    #                     help='Number of steps to run validation.')
    # parser.add_argument('--logger_name', default='./runs/runX/log',
    #                     help='Path to save Tensorboard log.')
    # parser.add_argument('--model_name', default='./runs/runX/checkpoint',
    #                     help='Path to save the model.')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('--max_violation', default=True, action='store_true',
    #                     help='Use max instead of sum in the rank loss.')
    # parser.add_argument('--img_dim', default=2048, type=int,
    #                     help='Dimensionality of the image embedding.')
    # parser.add_argument('--final_dims', default=256, type=int,
    #                     help='dimension of final codes.')
    # parser.add_argument('--max_words', default=32, type=int,
    #                     help='maximum number of words in a sentence.')
    # parser.add_argument("--bert_path",
    #                     default='/workspace/Image-Text_Retrieval/datasets/BERT_modal/uncased_L-12_H-768_A-12/',
    #                     type=str,
    #                     help="The BERT model path.")
    # parser.add_argument("--txt_stru", default='cnn',
    #                     help="Whether to use pooling or cnn or rnn")
    # parser.add_argument("--trans_cfg", default='t_cfg.json',
    #                     help="config file for image transformer")

    opt = parser.parse_args()
    model_path = '/workspace/Image-Text_Retrieval/saem/runs/coco_saem_nh/checkpoint_100/model_best.pth.tar'
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # opt.logger_name = opt.logger_name + TIMESTAMP
    # tb_logger.configure(opt.logger_name, flush_secs=5)

    # f = open(opt.logger_name+"opt.txt", 'w')
    # f.write(opt.__str__())
    # f.close()

    # opt.vocab_file = opt.bert_path + 'vocab.txt'
    # opt.bert_config_file = opt.bert_path + 'bert_config.json'
    # opt.init_checkpoint = opt.bert_path + 'pytorch_model.bin'
    # opt.do_lower_case = True

    # Load data loaders
    test_loader = data.get_test_loader('testall',
        opt.data_name, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SAEM(opt)

    opt.resume = '/workspace/Image-Text_Retrieval/saem/runs/coco_saem_nh/checkpoint_100/model_best.pth.tar'

    # optionally resume from a checkpoint
    # opt.resume = 'runs/f30k/log/checkpoint_27.pth.tar'
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        # Eiters is used to show logs as the continuation of another
        # training
        model.Eiters = checkpoint['Eiters']
        print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
              .format(opt.resume, start_epoch, best_rsum))
        validate(opt, test_loader, model)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    start = time.time()
    img_embs, cap_embs, cap_lens = encode_data(
        model, val_loader, opt, opt.log_step, logging.info)
    end = time.time()
    print("calculate backbone time:", end-start)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = 1-cdist(img_embs, cap_embs, metric='cosine')
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    print('r1', r1)
    print('r5', r5)
    print('r10', r10)
    print('medr', medr)
    print('meanr', meanr)
    print('r1i', r1i)
    print('r5i', r5i)
    print('r10i', r10i)
    print('medri', medri)
    print('meanr', meanr)
    print('rsum', currscore)

    return currscore


if __name__ == '__main__':
    main()
