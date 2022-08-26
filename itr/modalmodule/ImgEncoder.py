import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
import numpy as np
from collections import OrderedDict

from . import bert
from .utils import l1norm, l2norm


# VSE++|VSRN
class EncoderImageFull(nn.Module):

	def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
				 use_abs=False, no_imgnorm=False):
		"""Load pretrained VGG19 and replace top fc layer."""
		super(EncoderImageFull, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.use_abs = use_abs

		# Load a pre-trained model
		self.cnn = self.get_cnn(cnn_type, True)

		# For efficient memory usage.
		for param in self.cnn.parameters():
			param.requires_grad = finetune

		# Replace the last fully connected layer of CNN with a new one
		if cnn_type.startswith('vgg'):
			self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
								embed_size)
			self.cnn.classifier = nn.Sequential(
				*list(self.cnn.classifier.children())[:-1])
		elif cnn_type.startswith('resnet'):
			self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
			self.cnn.module.fc = nn.Sequential()

		self.init_weights()

	def get_cnn(self, arch, pretrained):
		"""Load a pretrained CNN and parallelize over GPUs
		"""
		if pretrained:
			print("=> using pre-trained model '{}'".format(arch))
			model = models.__dict__[arch](pretrained=True)
		else:
			print("=> creating model '{}'".format(arch))
			model = models.__dict__[arch]()

		if arch.startswith('alexnet') or arch.startswith('vgg'):
			model.features = nn.DataParallel(model.features)
			model.cuda()
		else:
			model = nn.DataParallel(model).cuda()

		return model

	def load_state_dict(self, state_dict):
		"""
		Handle the models saved before commit pytorch/vision@989d52a
		"""
		if 'cnn.classifier.1.weight' in state_dict:
			state_dict['cnn.classifier.0.weight'] = state_dict[
				'cnn.classifier.1.weight']
			del state_dict['cnn.classifier.1.weight']
			state_dict['cnn.classifier.0.bias'] = state_dict[
				'cnn.classifier.1.bias']
			del state_dict['cnn.classifier.1.bias']
			state_dict['cnn.classifier.3.weight'] = state_dict[
				'cnn.classifier.4.weight']
			del state_dict['cnn.classifier.4.weight']
			state_dict['cnn.classifier.3.bias'] = state_dict[
				'cnn.classifier.4.bias']
			del state_dict['cnn.classifier.4.bias']

		super(EncoderImageFull, self).load_state_dict(state_dict)

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""
		features = self.cnn(images)

		# normalization in the image embedding space
		features = l2norm(features)

		# linear projection to the joint embedding space
		features = self.fc(features)

		# normalization in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features)

		# take the absolute value of the embedding (used in order embeddings)
		if self.use_abs:
			features = torch.abs(features)

		return features


# VSE++|SCAN|SGRAF|VSRN
class EncoderImagePrecomp(nn.Module):
	def __init__(self, img_dim, embed_size, no_imgnorm=False, precomp_enc_type='basic', use_abs=False):
		super(EncoderImagePrecomp, self).__init__()
		self.use_abs = use_abs
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		if precomp_enc_type == 'basic':
			self.fc = nn.Linear(img_dim, embed_size)
			self.init_weights()
		elif precomp_enc_type == 'weight_norm':
			self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)
		else:
			raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""
		# assuming that the precomputed features are already l2-normalized

		features = self.fc(images)

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features, dim=-1)

		# take the absolute value of embedding (used in order embeddings)
		if self.use_abs:
			features = torch.abs(features)

		return features

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(EncoderImagePrecomp, self).load_state_dict(new_state)


# VSRN
from .vsrn_ import Rs_GCN


class EncoderImagePrecompAttn(nn.Module):
	def __init__(self, img_dim, embed_size, data_name, use_abs=False, no_imgnorm=False):
		super(EncoderImagePrecompAttn, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.use_abs = use_abs
		self.data_name = data_name

		self.fc = nn.Linear(img_dim, embed_size)
		self.init_weights()

		# GSR
		self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

		# GCN reasoning
		self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
		self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
		self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
		self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

		if self.data_name == 'f30k_precomp':
			self.bn = nn.BatchNorm1d(embed_size)

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""

		fc_img_emd = self.fc(images)
		if self.data_name != 'f30k_precomp':
			fc_img_emd = l2norm(fc_img_emd)

		# GCN reasoning
		# -> B,D,N
		GCN_img_emd = fc_img_emd.permute(0, 2, 1)
		GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
		GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
		GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
		GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
		# -> B,N,D
		GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

		GCN_img_emd = l2norm(GCN_img_emd)

		rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

		# features = torch.mean(rnn_img,dim=1)
		features = hidden_state[0]

		if self.data_name == 'f30k_precomp':
			features = self.bn(features)

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features)

		# take the absolute value of embedding (used in order embeddings)
		if self.use_abs:
			features = torch.abs(features)

		return features, GCN_img_emd

# def load_state_dict(self, state_dict):
# 	"""Copies parameters. overwritting the default one to
# 	accept state_dict from Full model
# 	"""
# 	own_state = self.state_dict()
# 	new_state = OrderedDict()
# 	for name, param in state_dict.items():
# 		if name in own_state:
# 			new_state[name] = param
#
# 	super(EncoderImagePrecompAttn, self).load_state_dict(new_state)


# SAEM
class FcMapping(nn.Module):
	"""
	MLP for image branch.
	"""

	def __init__(self, config):
		super(FcMapping, self).__init__()
		self.fc1 = nn.Linear(config['img_dim'], config['final_dims'])

	# self.fc2 = nn.Linear(config.final_dims*2, config.final_dims)

	def forward(self, x):
		# x: (batch_size, patch_num, img_dim)
		x = self.fc1(x)
		# x = F.relu(x)
		# x = self.fc2(x)
		embed = torch.mean(x, 1)  # (batch_size, final_dims)
		codes = F.normalize(embed, p=2, dim=1)
		return codes


class CnnMapping(nn.Module):
	def __init__(self, z, c):
		'''
		z: image patch dim
		c: final embedding dim
		'''
		super(CnnMapping, self).__init__()
		Co = 256  # number of channel for each kernel size
		Ks = [1, 2, 3]  # kernel size
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
			emb = (emb[:, :, :emb.size(2) / 2] + emb[:, :, emb.size(2) / 2:]) / 2

		embed = torch.mean(emb, 1)  # (batch_size, final_dims)
		codes = F.normalize(embed, p=2, dim=1)  # (N, C)
		return codes


class TransformerMapping(nn.Module):
	""" Self-attention layer for image branch
	"""

	def __init__(self, config):
		super(TransformerMapping, self).__init__()
		self.config = config
		bert_config = bert.BertConfig.from_json_file(config['trans_cfg'])
		self.layer = bert.BERTLayer(bert_config)
		self.mapping = nn.Linear(config['img_dim'], config['final_dims'])

	# self.mapping2 = nn.Linear(config['final_dims'], config['final_dims'])

	def forward(self, x):
		# x: (batch_size, patch_num, img_dim)
		x = self.mapping(x)  # x: (batch_size, patch_num, final_dims)
		attention_mask = torch.ones(x.size(0), x.size(1))
		if torch.cuda.is_available():
			attention_mask = attention_mask.cuda()
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		extended_attention_mask = extended_attention_mask.float()
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		hidden_states = self.layer(x, extended_attention_mask)
		# hidden_states = self.mapping2(hidden_states)
		embed = torch.mean(hidden_states, 1)  # (batch_size, final_dims)
		codes = F.normalize(embed, p=2, dim=1)  # (N, C)
		return codes


# CAMERA
from .camera_ import AGSA, Summarization
class EncoderImagePrecompSelfAttn(nn.Module):

	def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0):
		super(EncoderImagePrecompSelfAttn, self).__init__()
		self.embed_size = embed_size

		self.fc = nn.Linear(img_dim, embed_size)
		self.init_weights()
		self.position_enc = PositionEncoder(embed_size)
		self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
		self.mvs = Summarization(embed_size, smry_k)

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images, boxes, imgs_wh):
		"""Extract image feature vectors."""
		fc_img_emd = self.fc(images)
		fc_img_emd = l2norm(fc_img_emd)  # (bs, num_regions, dim)
		posi_emb = self.position_enc(boxes, imgs_wh)  # (bs, num_regions, num_regions, dim)

		# Adaptive Gating Self-Attention
		self_att_emb = self.agsa(fc_img_emd, posi_emb)  # (bs, num_regions, dim)
		self_att_emb = l2norm(self_att_emb)
		# Multi-View Summarization
		smry_mat = self.mvs(self_att_emb)
		L = F.softmax(smry_mat, dim=1)
		img_emb_mat = torch.matmul(L.transpose(1, 2), self_att_emb)  # (bs, k, dim)

		return F.normalize(img_emb_mat, dim=-1), smry_mat

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)


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
