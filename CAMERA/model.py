import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import torch.optim as optim

from models import TextEncoder
from models import PositionEncoder
from models import AGSA
from models import Summarization
from models import MultiViewMatching
from loss import TripletLoss, DiversityRegularization


def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class EncoderImagePrecompSelfAttn(nn.Module):

    def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0):
        super(EncoderImagePrecompSelfAttn, self).__init__()
        self.embed_size = embed_size

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        self.position_enc = PositionEncoder(embed_size)
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        self.mvs = Summarization(embed_size, smry_k)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, boxes, imgs_wh):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(fc_img_emd)  # (bs, num_regions, dim)
        posi_emb = self.position_enc(boxes, imgs_wh)  # (bs, num_regions, num_regions, dim)

        # Adaptive Gating Self-Attention
        self_att_emb = self.agsa(fc_img_emd, posi_emb)  # (bs, num_regions, dim)
        self_att_emb = l2norm(self_att_emb)
        # Multi-View Summarization
        smry_mat = self.mvs(self_att_emb)
        L = F.softmax(smry_mat, dim=1)
        img_emb_mat = torch.matmul(L.transpose(1, 2), self_att_emb)  # (bs, k, dim)

        return F.normalize(img_emb_mat, dim=-1), smry_mat

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)


class CAMERA(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecompSelfAttn(opt.img_dim, opt.embed_size,
                                                   opt.head, opt.smry_k, drop=opt.drop)
        self.txt_enc = TextEncoder(opt.bert_config_file, opt.init_checkpoint,
                                   opt.embed_size, opt.head, drop=opt.drop)

        if torch.cuda.is_available():
            self.img_enc = nn.DataParallel(self.img_enc)
            self.txt_enc = nn.DataParallel(self.txt_enc)
            self.img_enc.cuda()
            self.txt_enc.cuda()
            # cudnn.benchmark = True

        self.mvm = MultiViewMatching().cuda()
        # if torch.cuda.is_available():
        #     self.mvm = nn.DataParallel(self.mvm)
        #     self.mvm.cuda()
        # Loss and Optimizer
        self.crit_ranking = TripletLoss(margin=opt.margin, max_violation=opt.max_violation).cuda()
        self.crit_div = DiversityRegularization(opt.smry_k, opt.batch_size).cuda()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        # params = list(filter(lambda p: p.requires_grad, params))

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

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

    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, boxes, imgs_wh, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            boxes = boxes.cuda()
            imgs_wh = imgs_wh.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        # Forward
        cap_emb = self.txt_enc(input_ids, attention_mask, token_type_ids)  #  , lengths)
        img_emb, smry_mat = self.img_enc(images, boxes, imgs_wh)

        return img_emb, cap_emb, smry_mat

    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        self_att_emb, cap_emb, smry_mat = self.forward_emb(batch_data)
        bs = self_att_emb.size(0)
        # bidirectional triplet ranking loss
        sim_mat = self.mvm(self_att_emb, cap_emb)
        ranking_loss = self.crit_ranking(sim_mat)
        self.logger.update('Rank', ranking_loss.item(), bs)
        # diversity regularization
        div_reg = self.crit_div(smry_mat)
        self.logger.update('Div', div_reg.item(), bs)
        # total loss
        loss = ranking_loss + div_reg * self.opt.smry_lamda
        self.logger.update('Le', loss.item(), bs)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            if isinstance(self.params[0], dict):
                params = []
                for p in self.params:
                    params.extend(p['params'])
                clip_grad_norm_(params, self.grad_clip)
            else:
                clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()
