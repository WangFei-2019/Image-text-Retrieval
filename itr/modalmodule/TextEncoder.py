import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import l1norm, l2norm
from .bert import BertConfig, BertModel
from . import bert


# VSE++|SCAN|SGRAF|VSRN
class EncoderText(nn.Module):

	def __init__(self, vocab_size, word_dim, embed_size, num_layers,
				 use_bi_gru=False, no_txtnorm=False, dropout=0., use_abs=False, method_name=None):
		super(EncoderText, self).__init__()
		self.embed_size = embed_size
		self.no_txtnorm = no_txtnorm
		self.use_abs = use_abs
		self.method_name = method_name

		# word embedding
		self.embed = nn.Embedding(vocab_size, word_dim)
		self.dropout = nn.Dropout(dropout)

		# caption embedding
		self.use_bi_gru = use_bi_gru
		self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)

	def forward(self, x, lengths):
		"""Handles variable size captions
		"""
		# Embed word ids to vectors
		x = self.embed(x)
		x = self.dropout(x)

		# pack the caption
		packed = pack_padded_sequence(x, lengths, batch_first=True)

		# Forward propagate RNN
		out, _ = self.rnn(packed)

		# Reshape *final* output to (batch_size, hidden_size)
		cap_emb, cap_len = pad_packed_sequence(out, batch_first=True)

		if self.use_bi_gru:
			cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

		if self.method_name in {'VSE++', 'VSRN'}:
			I = torch.LongTensor(lengths).view(-1, 1, 1)
			I = torch.autograd.Variable(I.expand(x.size(0), 1, self.embed_size) - 1).cuda()
			cap_emb = torch.gather(cap_emb, 1, I).squeeze(1)

		# normalization in the joint embedding space
		if not self.no_txtnorm:
			cap_emb = l2norm(cap_emb, dim=-1)

		# take absolute value, used by order embeddings
		if self.use_abs:
			cap_emb = torch.abs(cap_emb)

		return cap_emb, cap_len


# SAEM
class BertMapping(nn.Module):
	"""
	"""

	def __init__(self, config):
		super(BertMapping, self).__init__()
		bert_config = BertConfig.from_json_file(config['bert_config_file'])
		self.bert = BertModel(bert_config)
		self.bert.load_state_dict(torch.load(config['init_checkpoint'], map_location='cpu'))
		self.freeze_layers(self.bert)

		self.txt_stru = config['txt_stru']

		if config['txt_stru'] == 'pooling':
			self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
			self.mapping_0 = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
			self.mapping = nn.Linear(bert_config.hidden_size, config['final_dims'])
		elif config['txt_stru'] == 'cnn':
			Ks = [1, 2, 3]
			in_channel = 1
			out_channel = 512
			embedding_dim = bert_config.hidden_size
			self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in Ks])
			self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
			self.mapping = nn.Linear(len(Ks) * out_channel, config['final_dims'])
		elif config['txt_stru'] == 'rnn':
			embedding_dim = bert_config.hidden_size
			self.bi_gru = config['bi_gru']
			self.rnn = nn.GRU(embedding_dim, config['embed_size'], config['num_layers'], batch_first=True,
							  bidirectional=config['bi_gru'])
			self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
			self.mapping = nn.Linear(config['embed_size'], config['final_dims'])
		elif config['txt_stru'] == 'trans':
			trans_config = BertConfig.from_json_file(config['trans_cfg'])
			self.layer = bert.BERTLayer(trans_config)
			self.dropout = nn.Dropout(trans_config.hidden_dropout_prob)
			self.mapping_0 = nn.Linear(bert_config.hidden_size, trans_config.hidden_size)
			self.mapping = nn.Linear(trans_config.hidden_size, config['final_dims'])
		else:
			raise ValueError("Unknown txt_stru: {}".format(config['txt_stru']))

	def forward(self, input_ids, attention_mask, token_type_ids, lengths):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
													  attention_mask=attention_mask)
		if self.txt_stru == 'pooling':
			output = self.mapping_0(all_encoder_layers[-1])
			output = torch.mean(output, 1)
		elif self.txt_stru == 'cnn':
			x = all_encoder_layers[-1].unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
			x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
			x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
			output = torch.cat(x, 1)
		elif self.txt_stru == 'rnn':
			x = all_encoder_layers[-1]  # (batch_size, token_num, embedding_dim)
			packed = pack_padded_sequence(x, lengths, batch_first=True)
			# Forward propagate RNN
			out, _ = self.rnn(packed)
			# Reshape *final* output to (batch_size, hidden_size)
			padded = pad_packed_sequence(out, batch_first=True)
			cap_emb, cap_len = padded
			if self.bi_gru:
				cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]) / 2
			else:
				cap_emb = cap_emb
			output = torch.mean(cap_emb, 1)
		elif self.txt_stru == 'trans':
			hidden_states = self.mapping_0(all_encoder_layers[-1])
			extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
			extended_attention_mask = extended_attention_mask.float()
			extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
			hidden_states = self.layer(hidden_states, extended_attention_mask)
			# output = hidden_states[:, 0, :]
			output = torch.mean(hidden_states, 1)

		output = self.dropout(output)
		code = self.mapping(output)
		# code = F.tanh(code)
		code = F.normalize(code, p=2, dim=1)
		return code

	def freeze_layers(self, model):
		for child in model.children():
			for param in child.parameters():
				param.requires_grad = False


# -------------------- CAMERA ----------------------------
from .camera_ import AGSA
class CAMERAEncoderText(nn.Module):
	"""
	"""
	def __init__(self, cfg_file, init_ckpt, embed_size, head, drop=0.0):
		super(CAMERAEncoderText, self).__init__()
		bert_config = BertConfig.from_json_file(cfg_file)
		self.bert = BertModel(bert_config)
		self.bert.load_state_dict(torch.load(init_ckpt, map_location='cpu'))
		self.freeze_layers(self.bert)

		self.mapping = nn.Linear(bert_config.hidden_size, embed_size)
		self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
		# MLP
		hidden_size = embed_size
		self.fc1 = nn.Linear(embed_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size)
		self.dropout = nn.Dropout(drop)

	def forward(self, input_ids, attention_mask, token_type_ids, lengths=None):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
													  attention_mask=attention_mask)
		x = self.mapping(all_encoder_layers[-1])  # (bs, token_num, final_dim)
		bs, token_num = x.size()[:2]
		agsa_emb = self.agsa(x)
		x = self.fc2(self.dropout(F.relu(self.fc1(agsa_emb))))
		x = (self.bn(x.view(bs * token_num, -1))).view(bs, token_num, -1)
		x = agsa_emb + self.dropout(x)  # context-enhanced word embeddings

		cap_emb = torch.mean(x, 1)
		return F.normalize(cap_emb, p=2, dim=-1)

	def freeze_layers(self, model):
		for child in model.children():
			for param in child.parameters():
				param.requires_grad = False
# ---------------------------------------------------------------------------------------