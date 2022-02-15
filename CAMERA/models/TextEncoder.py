import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
import torchvision as tv
from collections import OrderedDict 
import tokenization
from bert import BertModel, BertConfig
import bert
from models import AGSA

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class TextEncoder(nn.Module):
    """
    """
    def __init__(self, cfg_file, init_ckpt, embed_size, head, drop=0.0):
        super(TextEncoder, self).__init__()
        bert_config = BertConfig.from_json_file(cfg_file)
        self.bert = BertModel(bert_config)
        ckpt = torch.load(init_ckpt, map_location='cpu')
        self.bert.load_state_dict(ckpt)
        freeze_layers(self.bert)

        self.mapping = nn.Linear(bert_config.hidden_size, embed_size)
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        # MLP
        hidden_size = embed_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, input_ids, attention_mask, token_type_ids, lengths=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = self.mapping(all_encoder_layers[-1])    #(bs, token_num, final_dim)
        bs, token_num = x.size()[:2]
        agsa_emb = self.agsa(x)
        x = self.fc2(self.dropout(F.relu(self.fc1(agsa_emb))))
        x = (self.bn(x.view(bs*token_num, -1))).view(bs, token_num, -1)  
        x = agsa_emb + self.dropout(x)    # context-enhanced word embeddings

        cap_emb = torch.mean(x, 1)
        return F.normalize(cap_emb, p=2, dim=-1)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in own_state.items():
            if name in state_dict:
                new_state[name] = state_dict[name]
            else:
                new_state[name] = param

        super(TextEncoder, self).load_state_dict(new_state)
