import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import text_net
import loss
import image_net


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class SAEM(object):
    """
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.txt_enc = text_net.BertMapping(opt)
        self.img_enc = image_net.TransformerMapping(opt)
        # self.img_enc = image_net.RnnMapping(opt.img_dim, opt.final_dims, 1)
        # self.img_enc = image_net.CnnMapping(opt.img_dim, opt.final_dims)

        if torch.cuda.is_available():
            # self.img_enc = nn.DataParallel(self.img_enc)
            # self.txt_enc = nn.DataParallel(self.txt_enc)
            self.txt_enc.cuda()
            self.img_enc.cuda()
            # cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = loss.ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        self.criterion2 = loss.AngularLoss()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        # params = filter(lambda p: p.requires_grad, params)
        self.params = params
        _ = 0
        for i in params:
            _ += i.numel()
        print("Params numbers of whole model is ", _)

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def bert_data(self, images, input_ids, attention_mask, token_type_ids, lengths, ids):
        return images, input_ids, attention_mask, token_type_ids, lengths, ids

    def forward_emb(self, epoch, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, input_ids, attention_mask, token_type_ids, lengths, ids = self.bert_data(*batch_data)
        
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        # forward text
        cap_code = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths)
        cap_lens = lengths

        # forward image
        img_code = self.img_enc(images)

        return img_code, cap_code, cap_lens, ids

    def forward_loss(self, epoch, img_emb, cap_emb, cap_len, ids, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # alpha = 1
        if epoch > 20:
            alpha = 0
        else:
            alpha = 0.5 * (0.1 ** (epoch // 5))
        # alpha = 0
        loss1 = self.criterion(img_emb, cap_emb, cap_len, ids)
        loss2 = self.criterion2(img_emb, cap_emb, cap_len, ids)
        self.logger.update('Loss1', loss1.item(), img_emb.size(0))
        self.logger.update('Loss2', loss2.item(), img_emb.size(0))

        l2_reg = torch.tensor(0., dtype=torch.float)
        if torch.cuda.is_available():
            l2_reg = l2_reg.cuda()
        no_decay = ['bias', 'gamma', 'beta']
        for n, p in self.img_enc.named_parameters():
            en = n.split('.')[-1]
            if en not in no_decay:
                l2_reg += torch.norm(p)
        # for n, p in self.txt_enc.mapping.named_parameters():
        #     en = n.split('.')[-1]
        #     if en not in no_decay:
        #         l2_reg += torch.norm(p)
        # for n, p in self.txt_enc.layer.named_parameters():
        #     en = n.split('.')[-1]
        #     if en not in no_decay:
        #         l2_reg += torch.norm(p)
        reg_loss = 0.01 *l2_reg

        return loss1 + reg_loss + alpha*loss2
        # return loss2 + reg_loss

    def train_emb(self, epoch, batch_data, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, ids = self.forward_emb(epoch, batch_data)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_emb, cap_emb, cap_lens, ids)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
