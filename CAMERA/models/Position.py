import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init


def absoluteEncode(boxes, imgs_wh):
    # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, 2) '''
    x, y, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2] - boxes[:, :, 0], boxes[:, :, 3] - boxes[:, :, 1]
    expand_wh = torch.cat([imgs_wh, imgs_wh], dim=1).unsqueeze(dim=1)    #(bs, 1, 4)
    ratio_wh = (w / h).unsqueeze(dim=-1)  #(bs, num_r, 1)
    ratio_area = (w * h) / (imgs_wh[:, 0] * imgs_wh[:, 1]).unsqueeze(-1) #(bs, num_r)
    ratio_area = ratio_area.unsqueeze(-1) #(bs, num_r, 1)
    boxes = torch.stack([x, y, w, h], dim=2)
    boxes = boxes / expand_wh   #(bs, num_r, 4)
    res = torch.cat([boxes, ratio_wh, ratio_area], dim=-1)  #(bs, num_r, 6)
    return res

class PositionEncoder(nn.Module):
    '''Relative position Encoder
    '''
    def __init__(self, embed_dim, posi_dim=6):
        super(PositionEncoder, self).__init__()
        self.proj = nn.Linear(posi_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes, imgs_wh):
        # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, num_regions, 2)
        bs, num_regions = boxes.size()[:2]
        posi = absoluteEncode(boxes, imgs_wh)   #(bs, num_r, 4)

        x = self.proj(posi) 
        x = self.sigmoid(x)
        return x
