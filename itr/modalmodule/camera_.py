import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ------------------- AGSA -----------------------
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
		self.fc_g = nn.Linear(self.d_k, self.d_k * 2)

	def forward(self, inp, mask=None):
		nbatches = inp.size(0)
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (inp, inp, inp))]
		# gate
		G = self.fc_q(query) * self.fc_k(key)
		M = F.sigmoid(self.fc_g(G))  # (bs, h, num_region, d_k*2)
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
		x = self.att_layers[0](x, mask)  # (bs, r, d)
		x = (self.bns[0](x.view(bs * num_r, -1))).view(bs, num_r, -1)
		agsa_emb = rgn_emb + self.dropout[0](x)

		# 2nd~num_layers
		for i in range(self.num_layers - 1):
			x = self.att_layers[i + 1](agsa_emb, mask)  # (bs, r, d)
			x = (self.bns[i + 1](x.view(bs * num_r, -1))).view(bs, num_r, -1)
			agsa_emb = agsa_emb + self.dropout[i + 1](x)

		return agsa_emb


# ------------------- Summarizatiom -----------------------
class Summarization(nn.Module):
	''' Multi-View Summarization Module '''

	def __init__(self, embed_size, smry_k):
		super(Summarization, self).__init__()
		# dilation conv
		out_c = [256, 128, 128, 128, 128, 128, 128]
		k_size = [1, 3, 3, 3, 5, 5, 5]
		dila = [1, 1, 2, 3, 1, 2, 3]
		pads = [0, 1, 2, 3, 2, 4, 6]
		convs_dilate = [nn.Conv1d(embed_size, out_c[i], k_size[i], dilation=dila[i], padding=pads[i])
						for i in range(len(out_c))]
		self.convs_dilate = nn.ModuleList(convs_dilate)
		self.convs_fc = nn.Linear(1024, smry_k)

	def forward(self, rgn_emb):
		x = rgn_emb.transpose(1, 2)  # (bs, dim, num_r)
		x = [F.relu(conv(x)) for conv in self.convs_dilate]
		x = torch.cat(x, dim=1)  # (bs, 1024, num_r)
		x = x.transpose(1, 2)  # (bs, num_r, 1024)
		smry_mat = self.convs_fc(x)  # (bs, num_r, k)
		return smry_mat


# ------------------- Position -----------------------
def absoluteEncode(boxes, imgs_wh):
	# boxes -- (bs, num_regions, 4), imgs_wh -- (bs, 2) '''
	x, y, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2] - boxes[:, :, 0], boxes[:, :, 3] - boxes[:, :, 1]
	expand_wh = torch.cat([imgs_wh, imgs_wh], dim=1).unsqueeze(dim=1)  # (bs, 1, 4)
	ratio_wh = (w / h).unsqueeze(dim=-1)  # (bs, num_r, 1)
	ratio_area = (w * h) / (imgs_wh[:, 0] * imgs_wh[:, 1]).unsqueeze(-1)  # (bs, num_r)
	ratio_area = ratio_area.unsqueeze(-1)  # (bs, num_r, 1)
	boxes = torch.stack([x, y, w, h], dim=2)
	boxes = boxes / expand_wh  # (bs, num_r, 4)
	res = torch.cat([boxes, ratio_wh, ratio_area], dim=-1)  # (bs, num_r, 6)
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
		posi = absoluteEncode(boxes, imgs_wh)  # (bs, num_r, 4)

		x = self.proj(posi)
		x = self.sigmoid(x)
		return x

