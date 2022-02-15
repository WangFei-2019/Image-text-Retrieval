from __future__ import print_function
import os, sys
import pickle

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from model import CAMERA
from collections import OrderedDict
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            if(k == 'lr'):
                v = '{:.3e}'.format(v.val)
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    no_init = True
    for i, batch_data in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        ids = batch_data[5]
        # compute the embeddings
        img_emb, cap_emb, _ = model.forward_emb(batch_data, volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if no_init:
            no_init = False
            img = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)), dtype=np.float32)
            cap = np.zeros((len(data_loader.dataset), cap_emb.size(1)), dtype=np.float32)

        # preserve the embeddings by copying from gpu and converting to numpy
        img[ids] = img_emb.detach().cpu().numpy().copy()
        cap[ids] = cap_emb.data.cpu().numpy().copy()

        del batch_data

    return img, cap




def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = CAMERA(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])
    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)
    print('Computing results...')
    imgs, caps = encode_data(model, data_loader, opt)
    print('#Images: %d, #Captions: %d' %(imgs.shape[0] / 5, caps.shape[0]))

    if not fold5:
        imgs = numpy.array([imgs[i] for i in range(0, len(imgs), 5)])
        sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5)
        # no cross-validation, full evaluation
        r, rt = i2t(model, sims, return_ranks=True)
        ri, rti = t2i(model, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            imgs_block, caps_block = imgs[i * 5000:(i + 1) * 5000], caps[i * 5000:(i + 1) * 5000]
            imgs_block = numpy.array([imgs_block[i] for i in range(0, len(imgs_block), 5)])
            sims = calItr(model, imgs_block, caps_block, shard_size=opt.batch_size * 5)
            r, rt0, sims = i2t(sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(sims, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

def calItr(model, img_embs, cap_embs, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_img = len(img_embs)
    n_cap = len(cap_embs)


    t0 = time.time()
    n_im_shard = (n_img-1)//shard_size + 1
    n_cap_shard = (n_cap-1)//shard_size + 1
    d = np.zeros((n_img, n_cap))
    d_ids = np.zeros((n_img, n_cap))
    if sys.stdout.isatty():
        pbar = tqdm(total=(n_im_shard * n_cap_shard))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), n_img)
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), n_cap)
            with torch.no_grad():
                img_block = torch.from_numpy(img_embs[im_start:im_end]).cuda()
                cap_block = torch.from_numpy(cap_embs[cap_start:cap_end]).cuda()
                sim = model.mvm(img_block, cap_block)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
            if sys.stdout.isatty():
                pbar.update(1)
    if sys.stdout.isatty():
        pbar.close()
    print('Calculate similarity matrix elapses: {:.3f}s'.format(time.time() - t0))
    return d


def i2t(sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

