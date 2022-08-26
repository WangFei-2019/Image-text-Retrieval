import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .utils import l1norm, l2norm


# Calculate similarity
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim."""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def cosine_sim(im, s, *args):
	"""Cosine similarity between all the image and sentence pairs
	"""
	return im.mm(s.t())


def order_sim(im, s, *args):
	"""Order embeddings similarity measure $max(0, s-im)$
	"""
	YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
		   - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
	score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
	return score


# Loss function
class ContrastiveLoss(nn.Module):
	"""
	Compute contrastive loss
	"""

	def __init__(self, config, margin=0, measure=None, max_violation=False):
		super(ContrastiveLoss, self).__init__()
		self.config = config
		self.margin = margin
		self.max_violation = max_violation

		if measure == 'order':
			self.sim = order_sim
		elif measure == 'cosine':
			self.sim = cosine_sim
		else:
			raise ValueError("unknown measure:", measure)

		# SAEM
		if self.config['name'] == 'SAEM':
			if measure == 'order':
				self.sim = pdist
				print('pdist_order is used for similarity calculation')
			elif measure == 'cosine':
				self.sim = pdist_cos
				print('pdist_cosine is used for similarity calculation')
			else:
				raise ValueError("unknown measure:", measure)

		# SCAN
		elif self.config['name'] == 'SCAN':
			# compute image-sentence score matrix
			if self.config['cross_attn'] == 't2i':
				self.sim = xattn_score_t2i
			elif self.config['cross_attn'] == 'i2t':
				self.sim = xattn_score_i2t
			else:
				raise ValueError("unknown first norm type:", self.config['raw_feature_norm'])
		# SGRAF
		elif self.config['name'] == 'SGRAF':
			self.sim = lambda x, y, m, n: x

	def forward(self, im, s=None, s_l=None):
		# # SCAN
		# if self.config['name'] == 'SCAN':
		# 	# compute image-sentence score matrix
		# 	if self.config['cross_attn'] == 't2i':
		# 		scores = xattn_score_t2i(im, s, s_l, self.config)
		# 	elif self.config['cross_attn'] == 'i2t':
		# 		scores = xattn_score_i2t(im, s, s_l, self.config)
		# 	else:
		# 		raise ValueError("unknown first norm type:", self.config['raw_feature_norm'])
		# # SGRAF
		# elif self.config['name'] == 'SGRAF':
		# 	scores = im
		# # origin
		# else:
		scores = self.sim(im, s, s_l, self.config)

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
		mask = torch.eye(scores.size(0)) > .5
		if torch.cuda.is_available():
			I = mask.cuda()
		cost_s = cost_s.masked_fill_(I, 0)
		cost_im = cost_im.masked_fill_(I, 0)

		# keep the maximum violating negative for each query
		if self.max_violation:
			cost_s = cost_s.max(1)[0]
			cost_im = cost_im.max(0)[0]
		return cost_s.sum() + cost_im.sum()


##################################################

# VSRN
class RewardCriterion(nn.Module):
	def __init__(self):
		super(RewardCriterion, self).__init__()

	def forward(self, input, seq, reward):
		input = input.contiguous().view(-1)
		reward = reward.contiguous().view(-1)
		mask = (seq > 0).float()
		mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
						  mask[:, :-1]], 1).contiguous().view(-1)
		output = - input * reward * mask
		output = torch.sum(output) / torch.sum(mask)

		return output


# VSRN
class LanguageModelCriterion(nn.Module):
	def __init__(self):
		super(LanguageModelCriterion, self).__init__()
		self.loss_fn = nn.NLLLoss(reduce=False)

	def forward(self, logits, target, mask):
		"""
		logits: shape of (N, seq_len, vocab_size)
		target: shape of (N, seq_len)
		mask: shape of (N, seq_len)
		"""
		# truncate to the same size
		batch_size = logits.shape[0]
		target = target[:, :logits.shape[1]]
		mask = mask[:, :logits.shape[1]]
		logits = logits.contiguous().view(-1, logits.shape[2])
		target = target.contiguous().view(-1)
		mask = mask.contiguous().view(-1)
		loss = self.loss_fn(logits, target)
		output = torch.sum(loss * mask) / batch_size
		return output


# -------------------------------------------------------

# SAEM
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
			negative = np.concatenate([n_pairs[:i], n_pairs[i + 1:]])
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
			loss = torch.mean(torch.log(1 + x))
		else:
			cost = x.max(2)[0]
			loss = torch.log(1 + cost).sum()
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


# SAEM
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
		target = torch.tensor(ids) / 5
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
			loss = torch.log(1 + torch.exp(cost)).sum()

		return loss


##################################################

# SAEM

def pdist(x1, x2, *args):
	"""
		compute euclidean distance between two tensors
		x1: Tensor of shape (h1, w)
		x2: Tensor of shape (h2, w)
		Return pairwise euclidean distance for each row vector in x1, x2 as
		a Tensor of shape (h1, h2)
	"""
	x1_square = torch.sum(x1 * x1, 1).view(-1, 1)
	x2_square = torch.sum(x2 * x2, 1).view(1, -1)
	return torch.sqrt(x1_square - 2 * torch.mm(x1, x2.transpose(0, 1)) + x2_square + 1e-4)


def pdist_cos(x1, x2, *args):
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


# -------------------------------------------------------

# SCAN
def xattn_score_t2i(images, captions, cap_lens, config):
	"""
	Images: (n_image, n_regions, d) matrix of images
	Captions: (n_caption, max_n_word, d) matrix of captions
	CapLens: (n_caption) array of caption lengths
	"""
	similarities = []
	n_image = images.size(0)
	n_caption = captions.size(0)
	for i in range(n_caption):
		# Get the i-th text description
		n_word = cap_lens[i]
		cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
		# --> (n_image, n_word, d)
		cap_i_expand = cap_i.repeat(n_image, 1, 1)
		"""
			word(query): (n_image, n_word, d)
			image(context): (n_image, n_regions, d)
			weiContext: (n_image, n_word, d)
			attn: (n_image, n_region, n_word)
		"""
		weiContext, attn = func_attention(cap_i_expand, images, config, smooth=config['lambda_softmax'])
		cap_i_expand = cap_i_expand.contiguous()
		weiContext = weiContext.contiguous()
		# (n_image, n_word)
		row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
		if config['agg_func'] == 'LogSumExp':
			row_sim.mul_(config['lambda_lse']).exp_()
			row_sim = row_sim.sum(dim=1, keepdim=True)
			row_sim = torch.log(row_sim) / config['lambda_lse']
		elif config['agg_func'] == 'Max':
			row_sim = row_sim.max(dim=1, keepdim=True)[0]
		elif config['agg_func'] == 'Sum':
			row_sim = row_sim.sum(dim=1, keepdim=True)
		elif config['agg_func'] == 'Mean':
			row_sim = row_sim.mean(dim=1, keepdim=True)
		else:
			raise ValueError("unknown aggfunc: {}".format(config['agg_func']))
		similarities.append(row_sim)

	# (n_image, n_caption)
	similarities = torch.cat(similarities, 1)

	return similarities


# SCAN
def xattn_score_i2t(images, captions, cap_lens, config):
	"""
	Images: (batch_size, n_regions, d) matrix of images
	Captions: (batch_size, max_n_words, d) matrix of captions
	CapLens: (batch_size) array of caption lengths
	"""
	similarities = []
	n_image = images.size(0)
	n_caption = captions.size(0)
	n_region = images.size(1)
	for i in range(n_caption):
		# Get the i-th text description
		n_word = cap_lens[i]
		cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
		# (n_image, n_word, d)
		cap_i_expand = cap_i.repeat(n_image, 1, 1)
		"""
			word(query): (n_image, n_word, d)
			image(context): (n_image, n_region, d)
			weiContext: (n_image, n_region, d)
			attn: (n_image, n_word, n_region)
		"""
		weiContext, attn = func_attention(images, cap_i_expand, config, smooth=config['lambda_softmax'])
		# (n_image, n_region)
		row_sim = cosine_similarity(images, weiContext, dim=2)
		if config['agg_func'] == 'LogSumExp':
			row_sim.mul_(config['lambda_lse']).exp_()
			row_sim = row_sim.sum(dim=1, keepdim=True)
			row_sim = torch.log(row_sim) / config['lambda_lse']
		elif config['agg_func'] == 'Max':
			row_sim = row_sim.max(dim=1, keepdim=True)[0]
		elif config['agg_func'] == 'Sum':
			row_sim = row_sim.sum(dim=1, keepdim=True)
		elif config['agg_func'] == 'Mean':
			row_sim = row_sim.mean(dim=1, keepdim=True)
		else:
			raise ValueError("unknown aggfunc: {}".format(config['agg_func']))
		similarities.append(row_sim)

	# (n_image, n_caption)
	similarities = torch.cat(similarities, 1)
	return similarities


# SCAN
def func_attention(query, context, config, smooth, eps=1e-8):
	"""
	query: (n_context, queryL, d)
	context: (n_context, sourceL, d)
	"""
	batch_size_q, queryL = query.size(0), query.size(1)
	batch_size, sourceL = context.size(0), context.size(1)

	# Get attention
	# --> (batch, d, queryL)
	queryT = torch.transpose(query, 1, 2)

	# (batch, sourceL, d)(batch, d, queryL)
	# --> (batch, sourceL, queryL)
	attn = torch.bmm(context, queryT)
	if config['raw_feature_norm'] == "softmax":
		# --> (batch*sourceL, queryL)
		attn = attn.view(batch_size * sourceL, queryL)
		attn = nn.Softmax()(attn)
		# --> (batch, sourceL, queryL)
		attn = attn.view(batch_size, sourceL, queryL)
	elif config['raw_feature_norm'] == "l2norm":
		attn = l2norm(attn, 2)
	elif config['raw_feature_norm'] == "clipped_l2norm":
		attn = nn.LeakyReLU(0.1)(attn)
		attn = l2norm(attn, 2)
	elif config['raw_feature_norm'] == "l1norm":
		attn = l1norm_d(attn, 2)
	elif config['raw_feature_norm'] == "clipped_l1norm":
		attn = nn.LeakyReLU(0.1)(attn)
		attn = l1norm_d(attn, 2)
	elif config['raw_feature_norm'] == "clipped":
		attn = nn.LeakyReLU(0.1)(attn)
	elif config['raw_feature_norm'] == "no_norm":
		pass
	else:
		raise ValueError("unknown first norm type:", config['raw_feature_norm'])
	# --> (batch, queryL, sourceL)
	attn = torch.transpose(attn, 1, 2).contiguous()
	# --> (batch*queryL, sourceL)
	attn = attn.view(batch_size * queryL, sourceL)
	attn = nn.Softmax(dim=1)(attn * smooth)
	# --> (batch, queryL, sourceL)
	attn = attn.view(batch_size, queryL, sourceL)
	# --> (batch, sourceL, queryL)
	attnT = torch.transpose(attn, 1, 2).contiguous()

	# --> (batch, d, sourceL)
	contextT = torch.transpose(context, 1, 2)
	# (batch x d x sourceL)(batch x sourceL x queryL)
	# --> (batch, d, queryL)
	weightedContext = torch.bmm(contextT, attnT)
	# --> (batch, queryL, d)
	weightedContext = torch.transpose(weightedContext, 1, 2)

	return weightedContext, attnT


#############################################

# CAMERA
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
		if torch.cuda.is_available():
			mask = mask.cuda()
		cost_s = cost_s.masked_fill_(mask, 0)
		cost_im = cost_im.masked_fill_(mask, 0)

		# keep the maximum violating negative for each query
		if self.max_violation:
			cost_s = cost_s.max(1)[0]
			cost_im = cost_im.max(0)[0]

		return cost_s.sum() + cost_im.sum()


# CAMERA
class DiversityRegularization(nn.Module):
	"""
	Compute diversity regularization
	"""

	def __init__(self, smry_k, batch_size):
		super(DiversityRegularization, self).__init__()
		self.smry_k = smry_k
		self.batch_size = batch_size
		self.I = torch.eye(smry_k).unsqueeze(0).repeat(batch_size, 1, 1).cuda()  # (bs, k, k)

	def forward(self, smry_mat):
		bs = smry_mat.size(0)
		smry_mat = F.normalize(smry_mat, dim=1)  # (bs, num_r, k)
		diversity_loss = torch.matmul(smry_mat.transpose(1, 2), smry_mat)  # (bs, k, k)
		if bs != self.batch_size:
			I = torch.eye(self.smry_k).unsqueeze(0).repeat(bs, 1, 1).cuda()
		else:
			I = self.I
		diversity_loss = diversity_loss - I
		diversity_loss = (diversity_loss ** 2).sum()
		return diversity_loss
