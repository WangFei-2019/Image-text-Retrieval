import pickle
import os, sys
import time
import shutil
import numpy as np
import torch
import random

import data
from model import CAMERA
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, calItr
import logging
import tensorboard_logger as tb_logger
import argparse
from misc.utils import print_options


def setup_seed(seed):  # 2022/01/20 增加随机种子，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    # ---------------------------------------------------------------#
    parser.add_argument('--seed', default=None, help='default: random')  # 2022/01/21 10:19 增加随机种子
    parser.add_argument('--cuda', default='0', help='default: random')  # 2022/01/21 10:19 增加GPU选取
    # ---------------------------------------------------------------#
    parser.add_argument('--data_path', default='/workspace/Image-Text_Retrieval/datasets', help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=2048, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float,
                        help='Decay coefficient for learning rate updating')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int, help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/run_CAMERA', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument("--bert_path",
                        default='/workspace/Image-Text_Retrieval/datasets/BERT_modal/uncased_L-12_H-768_A-12/',
                        type=str, help="The BERT model path.")
    parser.add_argument('--max_words', default=32, type=int, help='maximum number of words in a sentence.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout')
    parser.add_argument('--head', type=int, default=64, help='Number of heads in AGSA')
    parser.add_argument('--smry_k', type=int, default=12, help='Number of views in summarization module')
    parser.add_argument('--smry_lamda', type=float, default=0.01, help='Trade-off in summarization module')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda  # 2022/01/21 18:22 设置使用的GPU

    if opt.seed is None:  # 2022/01/20 19:28 增加随机种子为文件命名
        opt.seed = random.randint(0, 100000)
    opt.seed = int(opt.seed)

    setup_seed(opt.seed)  # 2022/01/20 增加随机种子，使得实验可复现

    # Load Vocabulary Wrapper
    opt.vocab_file = opt.bert_path + 'vocab.txt'
    opt.bert_config_file = opt.bert_path + 'bert_config.json'
    opt.init_checkpoint = opt.bert_path + 'pytorch_model.bin'
    opt.do_lower_case = True

    opt.logger_name = opt.logger_name + '_seed' + str(
        opt.seed)  # + '_' + time.strftime('%Y%m%d-%H-%M-%S', time.localtime())

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # save setting
    argsDict = opt.__dict__
    with open(os.path.join(opt.logger_name, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    print_options(opt)
    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = CAMERA(opt)
    start_epoch = 0

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            best_r1 = checkpoint['best_r1']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    best_r1 = 0

    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        best_rsum, best_r1 = train(opt, train_loader, model, epoch, val_loader, best_rsum, best_r1)
        # evaluate on validation set
        rsum, r1 = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        best_r1 = max(r1, best_r1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'best_r1': best_r1,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader, best_rsum, best_r1):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()

    for i, train_data in enumerate(train_loader):
        # if opt.reset_train:
        # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(epoch, train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            # evaluate on validation set
            rsum, r1 = validate(opt, val_loader, model)
            # model.train_start()
            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            best_r1 = max(r1, best_r1)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'best_r1': best_r1,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, prefix=opt.logger_name + '/')

    return best_rsum, best_r1


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    t0 = time.time()
    imgs, caps = encode_data(model, val_loader, opt.log_step, logging.info)
    imgs = np.array([imgs[i] for i in range(0, len(imgs), 5)])
    sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r1i + r5i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, r1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (opt.lr_decay_gamma ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
