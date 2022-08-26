import random
import os
import time

import numpy
import torch
import logging
import tensorboard_logger as tb_logger
from itr.datamodule.vocab import Vocabulary  # NOQA

from .metricmodule import evaluation as eval
from .metricmodule import second2DHM
from .config import load_hyperparams


# Random seed
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	numpy.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def adjust_learning_rate(_config, optimizer, epoch):
	"""
	Sets the learning rate to the initial LR
	decayed by 10 after _config['lr_update'] epoch
	"""
	lr = _config['learning_rate'] * (0.1 ** (epoch // _config['lr_update']))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def load_resume(models, _config, reload=False):
	if os.path.exists(_config['resume']):
		print("=> loading checkpoint '{}'".format(_config['resume']))
		checkpoint = torch.load(_config['resume'])
		start_epoch = checkpoint['epoch']
		best_rsum = checkpoint['best_rsum']
		best_r1 = checkpoint['best_r1']
	else:
		raise FileNotFoundError("=> no checkpoint is found at '{}'".format(_config['resume']))
	if reload:
		config = checkpoint['_config']
		for name in load_hyperparams:
			_config[name] = config[name]
	model = models.get_model(_config).cuda()
	model.load_state_dict(checkpoint['model'])
	model.Eiters = checkpoint['Eiters']
	print("=> loaded checkpoint '{}' (epoch {}, best_rsum {}, best_rl {})"
		  .format(_config['resume'], start_epoch, best_rsum, best_r1))

	return model, start_epoch, best_rsum, best_r1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', is_epo_end=False):
	if is_epo_end:
		torch.save(state, os.path.join(prefix, 'epo' + str(state['epoch']) + '_' + filename))
	if is_best:
		torch.save(state, os.path.join(prefix, 'model_best.pth.tar'))


def print_options(config):
	print("")
	print("----- options -----".center(120, '-'))
	# config = vars(config)
	string = ''
	for i, (k, v) in enumerate(sorted(config.items())):
		string += "{}: {}".format(k, v).center(40, ' ')
		if i % 3 == 2 or i == len(config.items()) - 1:
			print(string)
			string = ''
	print("".center(120, '-'))
	print("")


def train_step(_config, train_loader, model, epoch, val_loader, best_rsum=0, best_r1=0):
	# average meters to record the training statistics
	batch_time = eval.AverageMeter()
	data_time = eval.AverageMeter()
	train_logger = eval.LogCollector()

	end = time.time()

	# switch to train mode
	model.train_start()

	for i, train_data in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end, n=1)

		# make sure train logger is used
		model.logger = train_logger

		# Update the model
		model.train_emb(train_data)

		# measure elapsed time
		batch_time.update(time.time() - end, n=1)
		end = time.time()

		# Print log info
		if model.Eiters % _config['log_step'] == 0:
			logging.info(
				'Epoch: [{0}][{1}/{2}]\t'
				'{e_log}\t'
				'Time {batch_time.avg:.3f} ({batch_time_sum})\t'
				'Data {data_time.avg:.3f} ({data_time_sum})\t'
					.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, e_log=str(model.logger),
					batch_time_sum=second2DHM(batch_time.sum)[0],
					data_time_sum=second2DHM(data_time.sum)[0]))

		# Record logs in tensorboard
		tb_logger.log_value('epoch', epoch, step=model.Eiters)
		tb_logger.log_value('step', i, step=model.Eiters)
		tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
		tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
		model.logger.tb_log(tb_logger, step=model.Eiters)

		# validate at every val_step
		if model.Eiters % _config['val_step'] == 0:
			rsum, r1 = validate_step(_config, val_loader, model)
			# remember best R@ sum and save checkpoint
			is_best = rsum > best_rsum
			best_rsum = max(rsum, best_rsum)
			best_r1 = max(r1, best_r1)
			save_checkpoint({
				'epoch': epoch,
				'model': model.state_dict(),
				'best_rsum': best_rsum,
				'best_r1': best_r1,
				'_config': _config,
				'Eiters': model.Eiters,
			}, is_best, prefix=_config['save_dir'])

			# switch to train mode
			model.train_start()


def validate_step(_config, val_loader, model):
	start = time.time()

	# switch to evaluate mode
	model.val_start()

	islength = True if _config['name'] in ['SGRAF', 'SCAN'] else False
	# compute the encoding for all the validation images and captions
	img_embs, cap_embs, cap_lens = eval.encode_data(model, val_loader, islength=islength)

	# clear duplicate 5*images and keep 1*images
	img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

	# record computation time of validation
	sims = eval.cal_sims(model, img_embs, cap_embs, lengths=cap_lens, shard_size=100)
	end = time.time()
	print("Calculate similarity time:", end - start)

	# caption retrieval
	(r1, r5, r10, medr, meanr) = eval.i2t(sims)
	logging.info("Image to text: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % (r1, r5, r10, medr, meanr))

	# image retrieval
	(r1i, r5i, r10i, medri, meanr) = eval.t2i(sims)
	logging.info("Text to image: r1 %.1f; r5 %.1f; r10 %.1f; medr %.1f; meanr %.1f" % (r1i, r5i, r10i, medri, meanr))

	# sum of recalls to be used for early stopping
	r_sum = r1 + r5 + r10 + r1i + r5i + r10i

	# record metrics in tensorboard
	tb_logger.log_value('r1_i2t', r1, step=model.Eiters)
	tb_logger.log_value('r5_i2t', r5, step=model.Eiters)
	tb_logger.log_value('r10_i2t', r10, step=model.Eiters)
	tb_logger.log_value('medr_i2t', medr, step=model.Eiters)
	tb_logger.log_value('meanr_i2t', meanr, step=model.Eiters)
	tb_logger.log_value('r1_t2i', r1i, step=model.Eiters)
	tb_logger.log_value('r5_t2i', r5i, step=model.Eiters)
	tb_logger.log_value('r10_t2i', r10i, step=model.Eiters)
	tb_logger.log_value('medr_t2i', medri, step=model.Eiters)
	tb_logger.log_value('meanr_t2i', meanr, step=model.Eiters)
	tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

	return r_sum, r1
