import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewMatching(nn.Module):
    def __init__(self, ):
        super(MultiViewMatching, self).__init__()

    def forward(self, imgs, caps):
        # caps -- (num_caps, dim), imgs -- (num_imgs, r, dim)
        num_caps  = caps.size(0)
        num_imgs, r = imgs.size()[:2]
        
        if num_caps == num_imgs:
            scores = torch.matmul(imgs, caps.t()) #(num_imgs, r, num_caps)
            scores = scores.max(1)[0]  #(num_imgs, num_caps)
        else:   
            scores = []
            score_ids = []
            for i in range(num_caps):
                cur_cap = caps[i].unsqueeze(0).unsqueeze(0)  #(1, 1, dim)
                cur_cap = cur_cap.expand(num_imgs, -1, -1)   #(num_imgs, 1, dim)
                cur_score = torch.matmul(cur_cap, imgs.transpose(-2, -1)).squeeze()    #(num_imgs, r)
                cur_score = cur_score.max(1, keepdim=True)[0]   #(num_imgs, 1)
                scores.append(cur_score)
            scores = torch.cat(scores, dim=1)   #(num_imgs, num_caps)

        return scores