import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

from . import ImgEncoder, TextEncoder, Objectives, Fusionmodule


# base module
class base_module(nn.Module):
	"""
	Base class of all methods.
	When a child class inherit the class, the variables of 'Build Models' should be define!
	"""

	def __init__(self, config):
		super(base_module, self).__init__()
		self.config = config
		self.grad_clip = config['grad_clip']
		self.Eiters = 0

		# Build Models
		self.img_enc = None
		self.txt_enc = None
		self.sim_enc = None
		self.criterion = None
		self.optimizer = None
		self.params = None

	def calculate_params(self):
		self.params_num = 0
		for i in self.params:
			self.params_num += i.numel()
		print("Optimizable parameter number of the whole model is ", self.params_num)

	def state_dict(self):
		state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()] if self.sim_enc is None \
			else [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc]
		return state_dict

	def load_state_dict(self, state_dict):
		self.img_enc.load_state_dict(state_dict[0])
		self.txt_enc.load_state_dict(state_dict[1])
		self.sim_enc.load_state_dict(state_dict[2]) if self.sim_enc is not None else None

	def train_start(self):
		"""switch to train mode
		"""
		self.img_enc.train()
		self.txt_enc.train()
		self.sim_enc.train() if self.sim_enc is not None else None

	def val_start(self):
		"""switch to evaluate mode
		"""
		self.img_enc.eval()
		self.txt_enc.eval()
		self.sim_enc.eval() if self.sim_enc is not None else None


# VSE++
class VSE_PP(base_module):
	"""
	rkiros/uvs model
	"""

	def __init__(self, config):
		super(VSE_PP, self).__init__(config)
		if config['data_name'].endswith('_precomp'):
			self.img_enc = ImgEncoder.EncoderImagePrecomp(config['img_dim'], config['embed_size'],
														  no_imgnorm=config['no_imgnorm'], precomp_enc_type='basic',
														  use_abs=config['use_abs'])
		else:
			self.img_enc = ImgEncoder.EncoderImageFull(config['embed_size'], config['finetune'], config['img_encoder'],
													   config['use_abs'], config['no_imgnorm'])
		self.txt_enc = TextEncoder.EncoderText(config['vocab_size'], config['word_dim'],
											   config['embed_size'], config['num_layers'],
											   use_abs=config['use_abs'], no_txtnorm=False)
		if torch.cuda.is_available():
			# self.img_enc = torch.nn.parallel.DataParallel(self.img_enc)
			# self.txt_enc = torch.nn.parallel.DataParallel(self.txt_enc)
			self.img_enc.cuda()
			self.txt_enc.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.criterion = Objectives.ContrastiveLoss(config=config, margin=config['margin'],
													max_violation=config['max_violation'], measure=config['measure'])

		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.fc.parameters())  # if use DP, change to 'list(self.img_enc.module.fc.parameters())'
		if config['finetune']:
			# if use DP, change to 'list(self.img_enc.module.cnn.parameters())'
			params += list(self.img_enc.cnn.parameters())
		self.params = params

		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

		self.calculate_params()

	def forward_emb(self, images, captions, lengths, *args, **kwargs):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward
		img_emb = self.img_enc(images)
		cap_emb, _ = self.txt_enc(captions, lengths)
		return img_emb, cap_emb

	def forward_loss(self, img_emb, cap_emb):
		"""
		Compute the loss given pairs of image and caption embeddings
		"""
		loss = self.criterion(img_emb, cap_emb)
		self.logger.update('Loss', loss.data, img_emb.size(0))
		return loss

	def train_emb(self, train_data):
		"""One training step given images and captions.
		"""

		images, _, _, captions, lengths, ids, _, _ = train_data

		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		img_emb, cap_emb = self.forward_emb(images, captions, lengths)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(img_emb, cap_emb)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm_(self.params, self.grad_clip)
		self.optimizer.step()


# SCAN
class SCAN(base_module):
	"""
	Stacked Cross Attention Network (SCAN) model
	"""

	def __init__(self, config):
		super(SCAN, self).__init__(config)
		# Build Models
		self.img_enc = ImgEncoder.EncoderImagePrecomp(config['img_dim'], config['embed_size'],
													  precomp_enc_type=config['precomp_enc_type'],
													  no_imgnorm=config['no_imgnorm'])
		self.txt_enc = TextEncoder.EncoderText(config['vocab_size'], config['word_dim'],
											   config['embed_size'], config['num_layers'],
											   use_bi_gru=config['bi_gru'],
											   no_txtnorm=config['no_txtnorm'])
		# Data Parallel
		if torch.cuda.is_available():
			# self.img_enc = torch.nn.parallel.DataParallel(self.img_enc)
			# self.txt_enc = torch.nn.parallel.DataParallel(self.txt_enc)
			self.img_enc.cuda()
			self.txt_enc.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.criterion = Objectives.ContrastiveLoss(config=config, margin=config['margin'],
													measure=config['measure'],
													max_violation=config['max_violation'])
		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.fc.parameters())
		self.params = params
		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

		self.calculate_params()

	def forward_emb(self, images, captions, lengths, *args, **kwargs):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward
		img_emb = self.img_enc(images)

		# cap_emb (tensor), cap_lens (list)
		cap_emb, cap_lens = self.txt_enc(captions, lengths)
		return img_emb, cap_emb, cap_lens

	def forward_loss(self, img_emb, cap_emb, cap_lens):
		"""
		Compute the loss given pairs of image and caption embeddings
		"""
		loss = self.criterion(img_emb, cap_emb, cap_lens)
		self.logger.update('Loss', loss.data, img_emb.size(0))
		return loss

	def train_emb(self, train_data):
		"""One training step given images and captions.
		"""
		images, _, _, captions, lengths, _, _, _ = train_data

		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(img_emb, cap_emb, cap_lens)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm_(self.params, self.grad_clip)
		self.optimizer.step()


# VSRN
class VSRN(base_module):
	"""
	rkiros/uvs model
	"""

	def __init__(self, config, use_txt_emb=True):
		super(VSRN, self).__init__(config=config)
		# Build Models
		if config['data_name'].endswith('_precomp'):
			if use_txt_emb == True:
				self.img_enc = ImgEncoder.EncoderImagePrecompAttn(
					config['img_dim'], config['embed_size'], config['data_name'], use_abs=config['use_abs'],
					no_imgnorm=config['no_imgnorm'])
			else:
				self.img_enc = ImgEncoder.EncoderImagePrecomp(
					config['img_dim'], config['embed_size'], use_abs=config['use_abs'], no_imgnorm=config['no_imgnorm'])
		else:
			self.img_enc = ImgEncoder.EncoderImageFull(
				config['embed_size'], config['finetune'], config['cnn_type'], use_abs=config['use_abs'],
				no_imgnorm=config['no_imgnorm'])

		self.txt_enc = TextEncoder.EncoderText(config['vocab_size'], config['word_dim'],
											   config['embed_size'], config['num_layers'],
											   use_abs=config['use_abs'], no_txtnorm=config['no_txtnorm'],
											   method_name=config['name'])

		# ###  captioning elements
		self.encoder = Fusionmodule.EncoderRNN(
			config['dim_vid'],
			config['dim_hidden'],
			bidirectional=config['bidirectional'],
			input_dropout_p=config['input_dropout_p'],
			rnn_cell=config['rnn_type'],
			rnn_dropout_p=config['rnn_dropout_p'])

		self.decoder = Fusionmodule.DecoderRNN(
			config['vocab_size'],
			config['max_len'],
			config['dim_hidden'],
			config['dim_word'],
			input_dropout_p=config['input_dropout_p'],
			rnn_cell=config['rnn_type'],
			rnn_dropout_p=config['rnn_dropout_p'],
			bidirectional=config['bidirectional'])

		self.caption_model = Fusionmodule.S2VTAttModel(self.encoder, self.decoder)

		if torch.cuda.is_available():
			# self.img_enc = torch.nn.parallel.DataParallel(self.img_enc)
			# self.txt_enc = torch.nn.parallel.DataParallel(self.txt_enc)
			# self.caption_model = torch.nn.parallel.DataParallel(self.caption_model)
			self.img_enc.cuda()
			self.txt_enc.cuda()
			self.caption_model.encoder.cuda()
			self.caption_model.decoder.cuda()
			self.caption_model.cuda()
			cudnn.benchmark = True

		self.crit = Objectives.LanguageModelCriterion()
		self.rl_crit = Objectives.RewardCriterion()

		# Loss and Optimizer
		self.criterion = Objectives.ContrastiveLoss(config=config,
													margin=config['margin'],
													measure=config['measure'],
													max_violation=config['max_violation'])
		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.parameters())
		params += list(self.caption_model.parameters())

		if config['finetune']:
			params += list(self.img_enc.cnn.parameters())
		self.params = params

		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
		self.calculate_params()

	def calcualte_caption_loss(self, fc_feats, labels, masks):
		torch.cuda.synchronize()
		labels = labels.cuda()
		masks = masks.cuda()

		seq_probs, _ = self.caption_model(fc_feats, labels, 'train')
		loss = self.crit(seq_probs, labels[:, 1:], masks[:, 1:])

		return loss

	def forward_emb(self, images, captions, lengths, *args, **kwargs):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward
		img_emb, GCN_img_emd = self.img_enc(images)
		cap_emb, _ = self.txt_enc(captions, lengths)
		return img_emb, cap_emb, GCN_img_emd

	def forward_loss(self, img_emb, cap_emb, GCN_img_emd, captions, captions_mask):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		# calcualte captioning loss
		caption_loss = self.calcualte_caption_loss(GCN_img_emd, captions, captions_mask)

		# measure accuracy and record loss
		retrieval_loss = self.criterion(img_emb, cap_emb)

		loss = retrieval_loss + caption_loss

		self.logger.update('Loss_caption', caption_loss.data, img_emb.size(0))
		self.logger.update('Loss_retrieval', retrieval_loss.data, img_emb.size(0))
		self.logger.update('Loss', loss.data, img_emb.size(0))
		return loss

	def train_emb(self, train_data):
		"""One training step given images and captions.
		"""
		images, _, _, captions, lengths, _, captions_mask, _ = train_data

		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		img_emb, cap_emb, GCN_img_emd = self.forward_emb(images, captions, lengths)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(img_emb, cap_emb, GCN_img_emd, captions, captions_mask)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm_(self.params, self.grad_clip)
		self.optimizer.step()


# SAEM
class SAEM(base_module):
	"""
	"""

	def __init__(self, config):
		super(SAEM, self).__init__(config=config)
		# Build Models
		self.img_enc = ImgEncoder.TransformerMapping(config)
		self.txt_enc = TextEncoder.BertMapping(config)

		if torch.cuda.is_available():
			# self.img_enc = nn.DataParallel(self.img_enc)
			# self.txt_enc = nn.DataParallel(self.txt_enc)
			self.txt_enc.cuda()
			self.img_enc.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.criterion = Objectives.ContrastiveLoss(config=config,
													margin=config['margin'],
													measure=config['measure'],
													max_violation=config['max_violation'])
		self.criterion_2 = Objectives.AngularLoss()

		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.parameters())
		self.params = params
		self.calculate_params()

		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

		self.no_decay = ['bias', 'gamma', 'beta']

	def forward_emb(self, images, captions, captions_mask, captions_type_ids, lengths, *args, **kwargs):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()
			captions_mask = captions_mask.cuda()
			captions_type_ids = captions_type_ids.cuda()
		# forward text
		cap_embs = self.txt_enc(captions, captions_mask, captions_type_ids, lengths)

		# forward image
		img_embs = self.img_enc(images)

		return img_embs, cap_embs

	def forward_loss(self, epoch, img_emb, cap_emb, cap_len, ids):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		# alpha = 1
		if epoch > 20:
			alpha = 0
		else:
			alpha = 0.5 * (0.1 ** (epoch // 5))
		# alpha = 0
		loss1 = self.criterion(img_emb, cap_emb, cap_len)
		loss2 = self.criterion_2(img_emb, cap_emb, cap_len, ids)
		self.logger.update('Loss1', loss1.item(), img_emb.size(0))
		self.logger.update('Loss2', loss2.item(), img_emb.size(0))

		l2_reg = torch.tensor(0., dtype=torch.float)
		if torch.cuda.is_available():
			l2_reg = l2_reg.cuda()
		for name, param in self.img_enc.named_parameters():
			if name.split('.')[-1] not in self.no_decay:
				l2_reg += torch.norm(param)
		reg_loss = 0.01 * l2_reg

		# return loss2 + reg_loss
		return loss1 + alpha * loss2 + reg_loss

	def train_emb(self, train_data, epoch=0):
		"""One training step given images and captions.
		"""
		images, _, _, captions, lengths, ids, captions_mask, captions_type_ids = train_data

		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		img_emb, cap_emb = self.forward_emb(images, captions, captions_mask, captions_type_ids, lengths)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(epoch, img_emb, cap_emb, lengths, ids)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm_(self.params, self.grad_clip)
		self.optimizer.step()


# SGRAF
class SGRAF(base_module):
	"""
	Similarity Reasoning and Filtration (SGRAF) Network
	"""

	def __init__(self, config):
		super(SGRAF, self).__init__(config)
		# Build Models
		self.img_enc = ImgEncoder.EncoderImagePrecomp(config['img_dim'], config['embed_size'],
													  no_imgnorm=config['no_imgnorm'], precomp_enc_type='basic')
		self.txt_enc = TextEncoder.EncoderText(config['vocab_size'], config['word_dim'],
											   config['embed_size'], config['num_layers'],
											   use_bi_gru=config['bi_gru'],
											   no_txtnorm=config['no_txtnorm'], dropout=.4)
		self.sim_enc = Fusionmodule.EncoderSimilarity(config['embed_size'], config['sim_dim'],
													  config['module_name'], config['sgr_step'])

		if torch.cuda.is_available():
			# self.img_enc = nn.DataParallel(self.img_enc)
			# self.txt_enc = nn.DataParallel(self.txt_enc)
			# self.sim_enc = nn.DataParallel(self.sim_enc)
			self.img_enc.cuda()
			self.txt_enc.cuda()
			self.sim_enc.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.criterion = Objectives.ContrastiveLoss(config=config, margin=config['margin'],
													measure=config['measure'],
													max_violation=config['max_violation'])
		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.parameters())
		params += list(self.sim_enc.parameters())
		self.params = params

		self.calculate_params()

		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

	def forward_emb(self, images, captions, lengths, *args, **kwargs):
		"""Compute the image and caption embeddings"""
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward feature encoding
		img_embs = self.img_enc(images)
		cap_embs, _ = self.txt_enc(captions, lengths)
		return img_embs, cap_embs

	def forward_loss(self, sims):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		loss = self.criterion(sims)
		self.logger.update('Loss', loss.item(), sims.size(0))
		return loss

	def train_emb(self, train_data):
		"""One training step given images and captions.
		"""
		images, _, _, captions, lengths, ids, _, _ = train_data

		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		img_embs, cap_embs = self.forward_emb(images, captions, lengths)
		sims = self.sim_enc(img_embs, cap_embs, lengths)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(sims)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm_(self.params, self.grad_clip)
		self.optimizer.step()


# CAMERA
class CAMERA(base_module):
	def __init__(self, config):
		super(CAMERA, self).__init__(config)
		# Build Models
		self.img_enc = ImgEncoder.EncoderImagePrecompSelfAttn(config['img_dim'], config['embed_size'],
															  config['head'], config['smry_k'], drop=config['drop'])
		self.txt_enc = TextEncoder.CAMERAEncoderText(config['bert_config_file'], config['init_checkpoint'],
													 config['embed_size'], config['head'], drop=config['drop'])
		self.mvm = Fusionmodule.MultiViewMatching()

		if torch.cuda.is_available():
			self.img_enc = nn.DataParallel(self.img_enc)
			self.txt_enc = nn.DataParallel(self.txt_enc)
			self.img_enc.cuda()
			self.txt_enc.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.crit_ranking = Objectives.TripletLoss(margin=config['margin'],
												   max_violation=config['max_violation']).cuda()
		self.crit_div = Objectives.DiversityRegularization(config['smry_k'], config['batch_size']).cuda()

		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.parameters())

		self.params = params
		self.calculate_params()

		self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

	def forward_emb(self, images, boxes, imgs_wh, captions, captions_mask, captions_type_ids, *args, **kwargs):
		"""
		Compute the image and caption embeddings
		"""

		# Set mini-batch dataset
		if torch.cuda.is_available():
			images = images.cuda()
			boxes = boxes.cuda()
			imgs_wh = imgs_wh.cuda()
			captions = captions.cuda()
			captions_mask = captions_mask.cuda()
			captions_type_ids = captions_type_ids.cuda()

		# Forward
		cap_emb = self.txt_enc(captions, captions_mask, captions_type_ids)
		img_emb, smry_mat = self.img_enc(images, boxes, imgs_wh)

		return img_emb, cap_emb, smry_mat

	def forward_loss(self, sim_mat, smry_mat):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		ranking_loss = self.crit_ranking(sim_mat)
		self.logger.update('Rank_Loss', ranking_loss.item(), len(sim_mat))

		# diversity regularization
		div_reg = self.crit_div(smry_mat)
		self.logger.update('Div_loss', div_reg.item(), len(sim_mat))

		# total loss
		loss = ranking_loss + div_reg * self.config['smry_lamda']
		self.logger.update('Loss', loss.item(), len(sim_mat))
		return loss

	def train_emb(self, train_data):
		"""
		One training step given images and captions.
		"""
		images, boxes, imgs_wh, captions, _, _, captions_mask, captions_type_ids = train_data
		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

		# compute the embeddings
		self_att_emb, cap_emb, smry_mat = self.forward_emb(images, boxes, imgs_wh, captions, captions_mask,
														   captions_type_ids)
		# bidirectional triplet ranking loss
		sim_mat = self.mvm(self_att_emb, cap_emb)

		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(sim_mat, smry_mat)

		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			if isinstance(self.params[0], dict):
				params = []
				for p in self.params:
					params.extend(p['params'])
				clip_grad_norm_(params, self.grad_clip)
			else:
				clip_grad_norm_(self.params, self.grad_clip)

		self.optimizer.step()
