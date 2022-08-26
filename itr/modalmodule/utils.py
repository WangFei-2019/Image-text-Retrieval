import torch


def l1norm(X, dim=1, eps=1e-8):
	"""L1-normalize columns of X """
	norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
	X = torch.div(X, norm)
	return X


def l2norm(X, dim=1, eps=1e-8):
	"""L2-normalize columns of X """
	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X
