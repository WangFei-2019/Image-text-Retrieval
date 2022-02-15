from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class TripletLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class DiversityRegularization(nn.Module):
    """
    Compute diversity regularization
    """
    def __init__(self, smry_k, batch_size):
        super(DiversityRegularization, self).__init__()
        self.smry_k = smry_k
        self.batch_size = batch_size
        self.I = torch.eye(smry_k).unsqueeze(0).repeat(batch_size, 1, 1).cuda() #(bs, k, k)

    def forward(self, smry_mat):
        bs = smry_mat.size(0)
        smry_mat = F.normalize(smry_mat, dim=1)   #(bs, num_r, k)
        diversity_loss = torch.matmul(smry_mat.transpose(1, 2), smry_mat)   #(bs, k, k)
        if bs != self.batch_size:
            I = torch.eye(self.smry_k).unsqueeze(0).repeat(bs, 1, 1).cuda()
        else:
            I = self.I
        diversity_loss = diversity_loss - I
        diversity_loss = (diversity_loss ** 2).sum()
        return diversity_loss