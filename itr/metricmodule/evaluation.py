from __future__ import print_function
import time
import os, sys
import yaml

import torch
import numpy as np
from collections import OrderedDict
import tqdm

from ..modalmodule import get_model, Objectives
from .. import datamodule as data


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=0):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / (.0001 + self.count)

	def __str__(self):
		"""String representation for logging
		"""
		# for values that should be recorded exactly e.g. iteration number
		if self.count == 0:
			return str(self.val)
		# for stats
		return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
	"""A collection of logging objects that can change from train to val"""

	def __init__(self):
		# to keep the order of logged variables deterministic
		self.meters = OrderedDict()

	def update(self, k, v, n=0):
		# create a new meter if previously not recorded
		if k not in self.meters:
			self.meters[k] = AverageMeter()
		self.meters[k].update(v, n)

	def __str__(self):
		"""Concatenate the meters in one log line
		"""
		s = ''
		for i, (k, v) in enumerate(self.meters.items()):
			if i > 0:
				s += '  '
			if (k == 'lr'):
				v = '{:.3e}'.format(v.val)
			s += k + ' ' + str(v)
		return s

	def tb_log(self, tb_logger, prefix='', step=None):
		"""Log using tensorboard
		"""
		for k, v in self.meters.items():
			tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, islength=False):
	"""Encode all images and captions loadable by `data_loader`
	"""
	val_logger = LogCollector()

	# switch to evaluate mode
	model.val_start()

	# numpy array to keep all the embeddings
	max_n_word = 0
	no_init = True
	if islength:
		for (_, _, _, _, lengths_, _, _, _) in data_loader:
			max_n_word = max(max_n_word, lengths_[0])
	for i, batch_data in enumerate(data_loader):
		# make sure val logger is used
		model.logger = val_logger
		images, boxes, imgs_wh, captions, lengths, ids, captions_mask, captions_type_ids = batch_data

		# compute the embeddings
		with torch.no_grad():
			emd_list = model.forward_emb(images=images, boxes=boxes, imgs_wh=imgs_wh, captions=captions,
										 lengths=lengths,
										 ids=ids, captions_mask=captions_mask, captions_type_ids=captions_type_ids)
		img_emb, cap_emb = emd_list[0], emd_list[1]

		# initialize the numpy arrays given the size of the embeddings
		if no_init:
			no_init = False
			ima_size = [len(data_loader.dataset)] + list(img_emb.size()[1:])
			cap_size = [len(data_loader.dataset)] + list(cap_emb.size()[1:])

			if islength:
				cap_size[1] = max_n_word

			img_embs = np.zeros(ima_size, dtype=np.float32)
			cap_embs = np.zeros(cap_size, dtype=np.float32)
			cap_lens = np.zeros(len(data_loader.dataset), dtype=np.int32)

		# preserve the embeddings by copying from gpu and converting to numpy
		img_embs[ids] = img_emb.detach().cpu().numpy().copy()
		cap_embs[ids, :cap_emb.size(1)] = cap_emb.data.cpu().numpy().copy()
		cap_lens[ids] = lengths

		del batch_data

	return img_embs, cap_embs, cap_lens


def cal_sims(model, img_embs, cap_embs, lengths=None, shard_size=128):
	"""
	calculate sims
	"""
	if model.config['name'] in ['CAMERA']:
		cal_fun = model.mvm
	else:
		cal_fun = model.sim_enc if model.sim_enc is not None else model.criterion.sim

	n_img = len(img_embs)
	n_cap = len(cap_embs)

	t0 = time.time()

	n_im_shard = (n_img - 1) // shard_size + 1
	n_cap_shard = (n_cap - 1) // shard_size + 1
	d = np.zeros((n_img, n_cap))
	# d_ids = np.zeros((n_img, n_cap))
	for i in range(n_im_shard):
		im_start, im_end = shard_size * i, min(shard_size * (i + 1), n_img)
		for j in range(n_cap_shard):
			cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), n_cap)
			with torch.no_grad():
				img_block = torch.from_numpy(img_embs[im_start:im_end]).cuda()
				cap_block = torch.from_numpy(cap_embs[cap_start:cap_end]).cuda()
				sim = cal_fun(img_block, cap_block, lengths, model.config)
			d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

	print('Calculate similarity matrix elapses: {:.3f}s'.format(time.time() - t0))
	return d


def i2t(sims, return_ranks=False):
	"""
	Images->Text (Image Annotation)
	Images: (N, n_region, d) matrix of images
	Captions: (5N, max_n_word, d) matrix of captions
	CapLens: (5N) array of caption lengths
	sims: (N, 5N) matrix of similarity im-cap
	"""
	npts = sims.shape[0]

	ranks = np.zeros(npts)
	top1 = np.zeros(npts)
	for index in range(npts):
		inds = np.argsort(sims[index])[::-1]

		# Score
		rank = 1e20
		for i in range(5 * index, 5 * index + 5, 1):
			tmp = np.where(inds == i)[0][0]
			if tmp < rank:
				rank = tmp
		ranks[index] = rank
		top1[index] = inds[0]

	# Compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	meanr = ranks.mean() + 1
	if return_ranks:
		return (r1, r5, r10, medr, meanr), (ranks, top1)
	else:
		return (r1, r5, r10, medr, meanr)


def t2i(sims, return_ranks=False):
	"""
	Text->Images (Image Search)
	Images: (5N, n_region, d) matrix of images
	Captions: (5N, max_n_word, d) or (5N, d) matrix of captions
	CapLens: (5N) array of caption lengths
	sims: (N, 5N) matrix of similarity im-cap
	"""
	npts = sims.shape[0]
	ranks = np.zeros(5 * npts)
	top1 = np.zeros(5 * npts)

	# --> (5N(caption), N(image))
	sims = sims.T

	for index in range(npts):
		for i in range(5):
			inds = np.argsort(sims[5 * index + i])[::-1]
			ranks[5 * index + i] = np.where(inds == index)[0][0]
			top1[5 * index + i] = inds[0]

	# Compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	meanr = ranks.mean() + 1
	if return_ranks:
		return (r1, r5, r10, medr, meanr), (ranks, top1)
	else:
		return (r1, r5, r10, medr, meanr)


def cal_recall(sims):
	res_dic = {}
	res_dic['result'] = []
	# no cross-validation, full evaluation
	r, rt = i2t(sims, return_ranks=True)
	ri, rti = t2i(sims, return_ranks=True)
	ar = (r[0] + r[1] + r[2]) / 3
	ari = (ri[0] + ri[1] + ri[2]) / 3
	rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
	print("rsum: %.1f" % rsum)
	print("Average i2t Recall: %.1f" % ar)
	print("Image to text: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % r)
	print("Average t2i Recall: %.1f" % ari)
	print("Text to image: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % ri)
	res_dic['rsum'] = rsum
	res_dic['i2t_ave_r'] = ar
	res_dic['i2t_r1'] = r[0]
	res_dic['i2t_r5'] = r[1]
	res_dic['i2t_r10'] = r[2]
	res_dic['i2t_medr'] = r[3]
	res_dic['i2t_meanr'] = r[4]
	res_dic['i2t_ranks'] = rt[0]
	res_dic['i2t_top1'] = rt[1]

	res_dic['t2i_ave_r'] = ari
	res_dic['t2i_r1'] = ri[0]
	res_dic['t2i_r5'] = ri[1]
	res_dic['t2i_r10'] = ri[2]
	res_dic['t2i_medr'] = ri[3]
	res_dic['t2i_meanr'] = ri[4]
	res_dic['t2i_ranks'] = rti[0]
	res_dic['t2i_top1'] = rti[1]

	res_dic['result'] = [list(r) + list(ri) + [ar, ari, rsum]]
	return res_dic


def evalrank_single(model_path, data_path=None, split='dev', fold5=False):
	"""
	Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
	cross-validation is done (only for MSCOCO). Otherwise, the full data is
	used for evaluation.
	"""
	# load model and options
	checkpoint = torch.load(model_path)
	_config = checkpoint['_config']
	print('Best model: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}' \
		  .format(checkpoint['epoch'], checkpoint['Eiters'], checkpoint['best_rsum'], checkpoint['best_r1']))
	if data_path is not None:
		_config['data_path'] = data_path

	# construct model
	model = get_model(_config)
	# load model state
	model.load_state_dict(checkpoint['model'])
	print(f'Loading dataset : {_config["data_name"]} ......')
	data_loader, _ = data.get_test_loader(split, _config['data_name'], _config['batch_size'], _config['workers'],
										  _config)
	print('Computing results...')
	islength = True if _config['name'] in ['SGRAF'] else False
	imgs_embs, caps_embs, cap_lens = encode_data(model, data_loader, islength=islength)
	print('#Images: %d, #Captions: %d' % (imgs_embs.shape[0] / 5, caps_embs.shape[0]))

	if not fold5:
		imgs_embs = np.array([imgs_embs[i] for i in range(0, len(imgs_embs), 5)])
		sims = cal_sims(model, imgs_embs, caps_embs, lengths=cap_lens, shard_size=_config['batch_size'] * 5)
		res_dic = cal_recall(sims)
		res_dic['data_name'] = _config['data_name']
	else:
		# 5fold cross-validation, only for MSCOCO
		res_dic = {}
		res_dic['sum_result'] = []
		for i in range(5):
			imgs_block = imgs_embs[i * 5000:(i + 1) * 5000:5]
			caps_block = caps_embs[i * 5000:(i + 1) * 5000]
			cap_lens_block = cap_lens[i * 5000:(i + 1) * 5000]
			sims = cal_sims(model, imgs_block, caps_block, lengths=cap_lens_block, shard_size=_config['batch_size'] * 5)
			print(f"--------------------- The {i + 1} part ---------------------")
			res_dic_ = cal_recall(sims)
			res_dic[f'PART_{i + 1}'] = res_dic_
			res_dic['sum_result'] += res_dic_['result']

		print("---------------------------------------------------------")
		print("--------------------- Mean metrics: ---------------------")
		mean_metrics = tuple(np.array(res_dic['sum_result']).mean(axis=0).flatten())
		print("rsum: %.1f" % (mean_metrics[10] * 6))
		print("Average i2t Recall: %.1f" % mean_metrics[11])
		print("Image to text: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % mean_metrics[:5])
		print("Average t2i Recall: %.1f" % mean_metrics[12])
		print("Text to image: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % mean_metrics[5:10])
		res_dic_['rsum'] = mean_metrics[10] * 6
		res_dic_['i2t_ave_r'] = mean_metrics[11]
		res_dic_['i2t_r1'] = mean_metrics[0]
		res_dic_['i2t_r5'] = mean_metrics[1]
		res_dic_['i2t_r10'] = mean_metrics[2]
		res_dic_['i2t_medr'] = mean_metrics[3]
		res_dic_['i2t_meanr'] = mean_metrics[4]

		res_dic_['t2i_ave_r'] = mean_metrics[12]
		res_dic_['t2i_r1'] = mean_metrics[5]
		res_dic_['t2i_r5'] = mean_metrics[6]
		res_dic_['t2i_r10'] = mean_metrics[7]
		res_dic_['t2i_medr'] = mean_metrics[8]
		res_dic_['t2i_meanr'] = mean_metrics[9]

		res_dic['Mean_metrics'] = res_dic_
		res_dic['data_name'] = _config['data_name'] + '_5fold'

	save_dir = '/'.join(model_path.split('/')[:-1])
	with open(os.path.join(save_dir, f'{res_dic["data_name"]}_single_result.yaml'), 'w') as yaml_file:
		yaml.dump(res_dic, yaml_file)


def evalrank_ensemble(model_path, model_path2, data_path=None, split='dev', fold5=False):
	"""
	Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
	cross-validation is done (only for MSCOCO). Otherwise, the full data is
	used for evaluation.
	"""
	# load model and options
	checkpoint = torch.load(model_path)
	_config = checkpoint['_config']
	model = get_model(_config)
	print('Best model 1: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}'
		  .format(checkpoint['epoch'], checkpoint['Eiters'], checkpoint['best_rsum'], checkpoint['best_r1']))

	checkpoint2 = torch.load(model_path2)
	_config_2 = checkpoint2['_config']
	model_2 = get_model(_config_2)
	print('Best model 2: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}'
		  .format(checkpoint2['epoch'], checkpoint2['Eiters'], checkpoint2['best_rsum'], checkpoint2['best_r1']))

	if data_path is not None:
		_config['data_path'] = data_path

	# load model state
	model.load_state_dict(checkpoint['model'])
	model_2.load_state_dict(checkpoint2['model'])

	print(f'Loading dataset ({_config["data_name"]})')
	data_loader, _ = data.get_test_loader(split, _config['data_name'], _config['batch_size'], _config['workers'],
										  _config)

	print('Computing results...')
	islength = True if _config['name'] in ['SGRAF'] else False
	imgs_embs, caps_embs, cap_lens = encode_data(model, data_loader, islength=islength)
	imgs_embs_2, caps_embs_2, cap_lens_2 = encode_data(model_2, data_loader, islength=islength)

	if not fold5:
		# no cross-validation, full evaluation
		imgs_embs = np.array([imgs_embs[i] for i in range(0, len(imgs_embs), 5)])
		imgs_embs_2 = np.array([imgs_embs_2[i] for i in range(0, len(imgs_embs_2), 5)])

		sims = cal_sims(model, imgs_embs, caps_embs, cap_lens, shard_size=_config['batch_size'] * 5)
		sims_2 = cal_sims(model_2, imgs_embs_2, caps_embs_2, cap_lens_2, shard_size=_config_2['batch_size'] * 5)
		sims = (sims + sims_2) / 2

		res_dic = cal_recall(sims)
		res_dic['data_name'] = _config['data_name']
	else:
		# 5fold cross-validation, only for MSCOCO
		res_dic = {}
		res_dic['sum_result'] = []
		for i in range(5):
			imgs_block, caps_block = imgs_embs[i * 5000:(i + 1) * 5000:5], caps_embs[i * 5000:(i + 1) * 5000]
			cap_lens_block = cap_lens[i * 5000:(i + 1) * 5000]

			imgs2_block, caps2_block = imgs_embs[i * 5000:(i + 1) * 5000:5], caps_embs[i * 5000:(i + 1) * 5000]
			cap_lens2_block = cap_lens[i * 5000:(i + 1) * 5000]

			sims = cal_sims(model, imgs_block, caps_block, cap_lens_block, shard_size=_config['batch_size'] * 5)
			sims_2 = cal_sims(model_2, imgs2_block, caps2_block, cap_lens2_block,
							  shard_size=_config_2['batch_size'] * 5)
			sims = (sims + sims_2) / 2

			print(f"--------------------- The {i + 1} part ---------------------")
			res_dic_ = cal_recall(sims)
			res_dic[f'PART_{i + 1}'] = res_dic_
			res_dic['sum_result'] += res_dic_['result']

		print("---------------------------------------------------------")
		print("--------------------- Mean metrics: ---------------------")
		mean_metrics = tuple(np.array(res_dic['sum_result']).mean(axis=0).flatten())
		print("rsum: %.1f" % (mean_metrics[10] * 6))
		print("Average i2t Recall: %.1f" % mean_metrics[11])
		print("Image to text: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % mean_metrics[:5])
		print("Average t2i Recall: %.1f" % mean_metrics[12])
		print("Text to image: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % mean_metrics[5:10])
		res_dic_['rsum'] = mean_metrics[10] * 6
		res_dic_['i2t_ave_r'] = mean_metrics[11]
		res_dic_['i2t_r1'] = mean_metrics[0]
		res_dic_['i2t_r5'] = mean_metrics[1]
		res_dic_['i2t_r10'] = mean_metrics[2]
		res_dic_['i2t_medr'] = mean_metrics[3]
		res_dic_['i2t_meanr'] = mean_metrics[4]

		res_dic_['t2i_ave_r'] = mean_metrics[12]
		res_dic_['t2i_r1'] = mean_metrics[5]
		res_dic_['t2i_r5'] = mean_metrics[6]
		res_dic_['t2i_r10'] = mean_metrics[7]
		res_dic_['t2i_medr'] = mean_metrics[8]
		res_dic_['t2i_meanr'] = mean_metrics[9]

		res_dic['Mean_metrics'] = res_dic_
		res_dic['data_name'] = _config['data_name'] + '_5fold'
		res_dic['modal_path_1'] = model_path
		res_dic['modal_path_2'] = model_path2

	save_dir = '/'.join(model_path.split('/')[:-1])
	with open(os.path.join(save_dir, f'{res_dic["data_name"]}_ensemble_result.yaml'), 'w') as yaml_file:
		yaml.dump(res_dic, yaml_file)
