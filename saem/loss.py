import torch
import torch.nn as nn
import numpy as np

def pdist(x1, x2):
    """
        compute euclidean distance between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise euclidean distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = torch.sum(x1*x1, 1).view(-1, 1)
    x2_square = torch.sum(x2*x2, 1).view(1, -1)
    return torch.sqrt(x1_square - 2 * torch.mm(x1, x2.transpose(0, 1)) + x2_square + 1e-4)


def pdist_cos(x1, x2):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    res = torch.mm(x1_norm, x2_norm.transpose(0, 1))
    mask = torch.isnan(res)
    res[mask] = 0
    return res


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, ids):
        scores = pdist_cos(im, s)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02, max_violation=True):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.max_violation = max_violation

    def forward(self, im, s, s_l, ids):
        target = ids / 5
        n_negatives = self.get_n_pairs(target)

        loss_im = self.n_pair_loss(im, s, s[n_negatives])
        loss_s = self.n_pair_loss(s, im, im[n_negatives])

        losses = loss_im + loss_s

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tensor n_negatives (n, n-1)
        """
        n_pairs = np.arange(len(labels))
        n_negatives = []
        for i in range(len(labels)):
            negative = np.concatenate([n_pairs[:i], n_pairs[i+1:]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_negatives)

    def n_pair_loss(self, anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)

        if not self.max_violation:
            x = torch.sum(torch.exp(x), 2)  # (n, 1)
            loss = torch.mean(torch.log(1+x))
        else:
            cost = x.max(2)[0]
            loss = torch.log(1+cost).sum()
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


class AngularLoss(NPairLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2, max_violation=True):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.max_violation = max_violation

    def forward(self, im, s, s_l, ids):
        target = ids / 5
        n_negatives = self.get_n_pairs(target)

        loss_im = self.angular_loss(im, s, s[n_negatives])
        loss_s = self.angular_loss(s, im, im[n_negatives])

        losses = loss_im + loss_s

        return losses

    def angular_loss(self, anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        if not self.max_violation:
            # Preventing overflow
            with torch.no_grad():
                t = torch.max(x, dim=2)[0]

            x = torch.exp(x - t.unsqueeze(dim=1))
            x = torch.log(torch.exp(-t) + torch.sum(x, 2))
            loss = torch.mean(t + x)
        else:
            cost = x.max(2)[0]
            loss = torch.log(1+torch.exp(cost)).sum()

        return loss