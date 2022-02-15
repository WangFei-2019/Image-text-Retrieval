import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GatedQueryAttLayer(nn.Module):
    def __init__(self, embed_size, h, is_share, drop=None):
        super(GatedQueryAttLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            self.linears = clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

        self.fc_q = nn.Linear(self.d_k, self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.d_k)
        self.fc_g = nn.Linear(self.d_k, self.d_k*2)

    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]
        # gate
        G = self.fc_q(query) * self.fc_k(key)
        M = F.sigmoid(self.fc_g(G)) # (bs, h, num_region, d_k*2)
        query = query * M[:, :, :, :self.d_k]
        key = key * M[:, :, :, self.d_k:]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x

class AGSA(nn.Module):
    ''' Adaptive Gating Self-Attention module '''
    def __init__(self, num_layers, embed_size, h=1, is_share=False, drop=None):
        super(AGSA, self).__init__()
        self.num_layers = num_layers
        self.bns = clones(nn.BatchNorm1d(embed_size), num_layers)
        self.dropout = clones(nn.Dropout(drop), num_layers)
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.att_layers = clones(GatedQueryAttLayer(embed_size, h, is_share, drop=drop), num_layers)

    def forward(self, rgn_emb, pos_emb=None, mask=None):
        ''' imb_emb -- (bs, num_r, dim), pos_emb -- (bs, num_r, num_r, dim) '''
        bs, num_r, emb_dim = rgn_emb.size()
        if pos_emb is None:
            x = rgn_emb
        else:
            x = rgn_emb * pos_emb
        
        # 1st layer
        x = self.att_layers[0](x, mask)    #(bs, r, d)
        x = (self.bns[0](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
        agsa_emb = rgn_emb + self.dropout[0](x)

        # 2nd~num_layers
        for i in range(self.num_layers - 1):
            x = self.att_layers[i+1](agsa_emb, mask) #(bs, r, d)
            x = (self.bns[i+1](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
            agsa_emb = agsa_emb + self.dropout[i+1](x)

        return agsa_emb
