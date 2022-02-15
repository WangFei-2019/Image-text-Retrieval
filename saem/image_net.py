import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
from bert import BertConfig, BertModel
import bert


class CnnMapping(nn.Module):
    def __init__(self, z, c):
        '''
        z: image patch dim
        c: final embedding dim
        '''
        super(CnnMapping, self).__init__()
        Co = 256 # number of channel for each kernel size
        Ks = [1, 2, 3] # kernel size
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, z)) for K in Ks])
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size,...)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(len(Ks) * Co, c)

    def forward(self, x):
        # x: (batch_size, token_num, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        # x = self.dropout(x)  # (N, len(Ks)*Co)
        codes = F.normalize(self.fc1(x), p=2, dim=1)  # (N, C)
        return codes


class RnnMapping(nn.Module):

    def __init__(self, z, c, num_layers=1, use_bi_gru=True):
        '''
        z: image patch dim
        c: final embedding dim
        '''
        super(RnnMapping, self).__init__()
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(z, c, num_layers, batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x):
        lengths = [36] * x.size(0)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        emb, _ = padded

        if self.use_bi_gru:
            emb = (emb[:,:,:emb.size(2)/2] + emb[:,:,emb.size(2)/2:])/2

        embed = torch.mean(emb, 1) # (batch_size, final_dims)
        codes = F.normalize(embed, p=2, dim=1)  # (N, C)
        return codes


class TransformerMapping(nn.Module):
    """ Self-attention layer for image branch
    """
    def __init__(self, opt):
        super(TransformerMapping, self).__init__()
        self.opt = opt
        bert_config = BertConfig.from_json_file(opt.trans_cfg)
        self.layer = bert.BERTLayer(bert_config)
        self.mapping = nn.Linear(opt.img_dim, opt.final_dims)
        #self.mapping2 = nn.Linear(opt.final_dims, opt.final_dims)

    def forward(self, x):
        # x: (batch_size, patch_num, img_dim)
        x = self.mapping(x) # x: (batch_size, patch_num, final_dims)
        attention_mask = torch.ones(x.size(0), x.size(1))
        if torch.cuda.is_available():
            attention_mask = attention_mask.cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states = self.layer(x, extended_attention_mask)
        # hidden_states = self.mapping2(hidden_states)
        embed = torch.mean(hidden_states, 1) # (batch_size, final_dims)
        codes = F.normalize(embed, p=2, dim=1)  # (N, C)
        return codes


class FcMapping(nn.Module):
    """ MLP for image branch.
    """
    def __init__(self, opt):
        super(FcMapping, self).__init__()
        self.fc1 = nn.Linear(opt.img_dim, opt.final_dims)
        # self.fc2 = nn.Linear(opt.final_dims*2, opt.final_dims)

    def forward(self, x):
        # x: (batch_size, patch_num, img_dim)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        embed = torch.mean(x, 1)  # (batch_size, final_dims)
        codes = F.normalize(embed, p=2, dim=1)
        return codes
