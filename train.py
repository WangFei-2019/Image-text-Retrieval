import os
import copy

import logging
import tensorboard_logger as tb_logger

from itr.config import ex
from itr import utils
from itr import datamodule as data
from itr import modalmodule as models
from itr.datamodule.vocab import Vocabulary  # NOQA

import resource

os.environ["NCCL_DEBUG"] = "INFO"
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def train(_config):
	# read Hyperparameters
	_config = copy.deepcopy(_config)

	# set random seed
	utils.setup_seed(_config['seed'])

	# set log to save information
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
	tb_logger.configure(_config['save_dir'], flush_secs=5)

	utils.print_options(_config)

	# Load data loaders
	train_loader, val_loader, vocab_size = data.get_loaders(_config['data_name'], _config['batch_size'],
															_config['workers'], _config)
	_config['vocab_size'] = vocab_size

	# optionally resume from a checkpoint
	if _config['resume']:
		model, start_epoch, best_rsum, best_r1 = utils.load_resume(models, _config, reload=True)
		utils.validate_step(_config, val_loader, model)
	else:
		start_epoch = 0
		best_rsum = 0
		best_r1 = 0
		# Construct the model
		model = models.get_model(_config).cuda()

	# Train the Model
	for epoch in range(start_epoch, _config['num_epochs']):
		utils.adjust_learning_rate(_config, model.optimizer, epoch)

		# train for one epoch
		utils.train_step(_config, train_loader, model, epoch, val_loader, best_rsum, best_r1)

		# evaluate on validation set
		r1, r_sum = utils.validate_step(_config, val_loader, model)

		# remember best R@ sum and save checkpoint
		is_best = r_sum > best_rsum
		best_rsum = max(r_sum, best_rsum)
		best_r1 = max(r1, best_r1)

		utils.save_checkpoint({
			'epoch': epoch,
			'model': model.state_dict(),
			'best_rsum': best_rsum,
			'best_rl': best_r1,
			'_config': _config,
			'Eiters': model.Eiters,
		}, is_best, filename='checkpoint.pth.tar', prefix=_config['save_dir'], is_epo_end=True)
