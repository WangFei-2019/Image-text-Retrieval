import os
import time
import random
import yaml

from sacred import Experiment

ex = Experiment("ITR")

__all__ = ['VSE_PP', 'SCAN', 'VSRN', 'SAEM', 'SGRAF', 'CAMERA']

load_hyperparams = ['img_encoder', 'crop_size', 'img_dim', 'no_imgnorm', 'use_bbox', 'finetune', 'precomp_enc_type',
					'trans_cfg', 'head', 'text_encoder', 'bi_gru', 'word_dim', 'no_txtnorm', 'num_layers', 'max_words',
					'txt_stru', 'embed_size', 'measure', 'use_abs', 'final_dims', 'sim_dim', 'rnn_type',
					'bidirectional', 'dim_hidden', 'dim_vid', 'input_dropout_p', 'rnn_dropout_p', 'dim_word', 'max_len',
					'module_name', 'sgr_step', 'max_violation', 'margin', 'cross_attn', 'raw_feature_norm', 'agg_func',
					'lambda_lse', 'lambda_softmax', 'smry_k', 'smry_lamda', 'lr_decay_gamma', 'drop']


@ex.config
def config():
	name = 'ITR'

	# ## load and save
	data_path = "/workspace/dataset/data"
	data_name = "f30k_precomp"  # {coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k. coco|f8k|f30k only used in the VSE++.
	vocab_path = "./itr/vocab"  # vocabulary path
	vocab_type = 'json'  # The file saving Vocabulary information is end with 'json'|'pkl'.
	save_path = "./runs"
	tail = None  # Add str after name of experiment file name.

	# ### server setting
	seed = 0  # random seed
	cuda = "2"
	workers = 8  # Number of data loader workers.

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 30  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0002  # Initial learning rate.
	lr_update = 15  # Number of epochs to update the learning rate.
	val_step = 500  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.
	use_restval = False  # Use the restval data for training on MSCOCO.

	# ## image encoder setting
	img_encoder = 'vgg19'  # The CNN used for image encoder (e.g. vgg19, resnet152)
	crop_size = 224  # Size of an image crop as the CNN input.
	img_dim = 4096  # Dimensionality of the image embedding.
	no_imgnorm = False  # Do not normalize the image embeddings.
	use_bbox = False  # Use region feature
	finetune = False  # Fine-tune the image encoder.
	precomp_enc_type = "basic"  # The parameter initialization method of image encoder. 'basic'|'weight_norm'
	# ## ### only for a bert image encoder
	trans_cfg = './itr/trans_cfg.json'  # config file for image transformer
	# ## ### CAMERA, AGSA is a module in text/image encoder.
	head = 64  # Number of heads in AGSA

	# ## text encoder setting
	text_encoder = 'gru'  # The RNN used for text encoder. lstm|gru|bert
	bi_gru = False  # Use bidirectional GRU. When 'text_encoder' is 'bert', 'bi_gru' is a param for bert decoder feature.
	word_dim = 300  # Dimensionality of the word embedding.
	no_txtnorm = False  # Do not normalize the text embeddings.
	num_layers = 1  # Number of LSTM/GRU layers.
	# ## ### only for a bert text encoder
	bert_path = '/workspace/dataset/uncased_L-12_H-768_A-12'  # The BERT model path.
	max_words = 32  # maximum number of words in a sentence.
	txt_stru = 'cnn'  # Whether to use pooling or cnn or rnn behind bert. 'pooling'/'cnn'/'rnn'/'trans'

	# ## fusion module setting and final joint embedding size
	embed_size = 1024  # Dimensionality of the joint embedding.
	measure = 'cosine'  # Similarity measure used (cosine|order)
	use_abs = False  # Take the absolute value of embedding vectors.
	final_dims = 256  # The dimension of followed extra encoder codes. Some methods use an extra encoder follow text/image encoder.
	sim_dim = 256  # Dimensionality of the sim embedding.
	# ## ### In VSRN, the params is used in encoder-decoder module in fusion module.
	rnn_type = 'gru'  # The rnn type of Fusion module Encoder/Decoder structure. 'lstm'|'gru'
	bidirectional = 0  # 0 for disable, 1 for enable. encoder/decoder bidirectional.
	dim_hidden = 512  # size of the rnn hidden layer in Fusion module Encoder/Decoder structure.
	dim_vid = 2048  # dim of features of Encoder structure.
	input_dropout_p = 0.2  # strength of dropout in the Language Model RNN.
	rnn_dropout_p = 0.5  # strength of dropout in the Language Model RNN.
	dim_word = 300  # the encoding size of each token in the vocabulary.
	max_len = 60  # max length of captions(containing <sos>,<eos>). One of the rnn decoder use it.
	# ## ### Params used in SGRAF fusion module.
	module_name = 'SGR'  # the mode of SGRAF. SGR|SAF
	sgr_step = 3  # Step of the SGR.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.
	# ## ### attention in SCAN ContrastiveLoss
	cross_attn = "t2i"  # 't2i|i2t'
	raw_feature_norm = "clipped_l2norm"  # 'clipped_l2norm'|'l2norm'|'clipped_l1norm'|'l1norm'|'no_norm'|'softmax'
	agg_func = "LogSumExp"  # 'LogSumExp'|'Mean'|'Max'|'Sum'
	lambda_lse = 6  # LogSumExp temp.
	lambda_softmax = 9.  # Attention softmax temperature.
	# ## ### CAMERA
	smry_k = 12  # Number of views in summarization module
	smry_lamda = 0.01  # Trade-off in summarization module

	# CAMERA
	lr_decay_gamma = 0.1  # Decay coefficient for learning rate updating.
	drop = 0.0  # Dropout


@ex.named_config
def VSE_PP():  # BMVC 2017
	'''
	"VSE++: Improving Visual-Semantic Embeddings with Hard Negatives" is published on BMVC 2017. ("https://arxiv.org/abs/1707.05612").
	The code original url is "https://github.com/fartashf/vsepp".
	'''
	name = "VSE++"

	data_name = "f30k_precomp"  # {coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k. coco|f8k|f30k only used in the VSE++.

	vocab_type = 'pkl'  # The file saving Vocabulary information is end with 'json'|'pkl'.

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 30  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0002  # Initial learning rate.
	lr_update = 15  # Number of epochs to update the learning rate.
	val_step = 10  # 500  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.
	use_restval = False  # Use the restval data for training on MSCOCO.

	# ## image encoder setting
	img_encoder = 'vgg19'  # The CNN used for image encoder (e.g. fc, vgg19, resnet152)
	crop_size = 224  # Size of an image crop as the CNN input.
	img_dim = 4096  # Dimensionality of the image embedding.
	no_imgnorm = False  # Do not normalize the image embeddings.
	finetune = False  # Fine-tune the image encoder.

	# ## text encoder setting
	word_dim = 300  # Dimensionality of the word embedding.
	num_layers = 1  # Number of LSTM/GRU layers.
	no_txtnorm = True  # Do not normalize the text embeddings.

	# ## fusion module setting and final joint embedding size
	embed_size = 1024  # Dimensionality of the joint embedding.
	measure = 'cosine'  # Similarity measure used (cosine|order)
	use_abs = False  # Take the absolute value of embedding vectors.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.


@ex.named_config
def SCAN():  # ECCV 2018
	'''
	"Stacked Cross Attention for Image-Text Matching" is published on ECCV 2018. ("https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf"),
	The code original url is "https://github.com/kuanghuei/SCAN".
	'''
	name = "SCAN"

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 30  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0002  # Initial learning rate.
	lr_update = 15  # Number of epochs to update the learning rate.
	val_step = 500  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.

	# ## image encoder setting
	img_dim = 2048  # Dimensionality of the image embedding.
	no_imgnorm = False  # Do not normalize the image embeddings.
	precomp_enc_type = "basic"  # The parameter initialization method of image encoder. 'basic'|'weight_norm'

	# ## text encoder setting
	text_encoder = 'gru'  # The RNN used for text encoder. lstm|gru|bert
	bi_gru = False  # Use bidirectional GRU. When 'text_encoder' is 'bert', 'bi_gru' is a param for bert decoder feature.
	word_dim = 300  # Dimensionality of the word embedding.
	no_txtnorm = True  # Do not normalize the text embeddings.
	num_layers = 1  # Number of LSTM/GRU layers.

	# ## fusion module setting and final joint embedding size
	embed_size = 1024  # Dimensionality of the joint embedding.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.
	# ## ### attention in SCAN ContrastiveLoss
	cross_attn = "t2i"  # 't2i|i2t'
	raw_feature_norm = "clipped_l2norm"  # 'clipped_l2norm'|'l2norm'|'clipped_l1norm'|'l1norm'|'no_norm'|'softmax'
	agg_func = "LogSumExp"  # 'LogSumExp'|'Mean'|'Max'|'Sum'
	lambda_lse = 6  # LogSumExp temp.
	lambda_softmax = 9.  # Attention softmax temperature.


@ex.named_config
def VSRN():  # ECCV 2018
	'''
	"Stacked Cross Attention for Image-Text Matching" is published on ECCV 2018. ("https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf"),
	The code original url is "https://github.com/kuanghuei/SCAN".
	'''
	name = "VSRN"

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 30  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0002  # Initial learning rate.
	lr_update = 15  # Number of epochs to update the learning rate.
	val_step = 500  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.
	use_restval = False  # Use the restval data for training on MSCOCO.

	# ## image encoder setting
	img_encoder = 'vgg19'  # The CNN used for image encoder (e.g. vgg19, resnet152)
	crop_size = 224  # Size of an image crop as the CNN input.
	img_dim = 2048  # Dimensionality of the image embedding.
	no_imgnorm = False  # Do not normalize the image embeddings.
	use_bbox = False  # Use region feature
	finetune = False  # Fine-tune the image encoder.
	precomp_enc_type = "basic"  # The parameter initialization method of image encoder. 'basic'|'weight_norm'
	# ## ### only for a bert image encoder
	trans_cfg = './itr/trans_cfg.json'  # config file for image transformer

	# ## text encoder setting
	text_encoder = 'gru'  # The RNN used for text encoder. lstm|gru|bert
	bi_gru = False  # Use bidirectional GRU. When 'text_encoder' is 'bert', 'bi_gru' is a param for bert decoder feature.
	word_dim = 300  # Dimensionality of the word embedding.
	no_txtnorm = False  # Do not normalize the text embeddings.
	num_layers = 1  # Number of LSTM/GRU layers.

	# ## fusion module setting and final joint embedding size
	embed_size = 2048  # Dimensionality of the joint embedding.
	measure = 'cosine'  # Similarity measure used (cosine|order)
	use_abs = False  # Take the absolute value of embedding vectors.
	# ## ### In VSRN, the params is used in encoder-decoder module in fusion module.
	rnn_type = 'gru'  # The rnn type of Fusion module Encoder/Decoder structure. 'lstm'|'gru'
	bidirectional = False  # False for disable, True for enable. encoder/decoder bidirectional.
	dim_hidden = 512  # # size of the rnn hidden layer in Fusion module Encoder/Decoder structure.
	dim_vid = 2048  # dim of features of Encoder structure.
	input_dropout_p = 0.2  # strength of dropout in the Language Model RNN.
	rnn_dropout_p = 0.5  # strength of dropout in the Language Model RNN.
	dim_word = 300  # the encoding size of each token in the vocabulary.
	max_len = 60  # max length of captions(containing <sos>,<eos>). One of the rnn decoder use it.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.


@ex.named_config
def SAEM():  # ACM MM 2019
	'''
	"Learning Fragment Self-Atention Embeddings for Image-Text Matching" is published on ACMM MM 2022. ("http://www.jdl.link/doc/2011/20191229_ACMMMLearning%20Fragment%20Self-Attention%20Embeddings%20for%20Image-Text%20Matching.pdf"),
	The code original url is "https://github.com/yiling2018/saem".
	'''
	name = "SAEM"

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 30  # Number of training epochs.
	batch_size = 64  # Size of a training mini-batch.
	learning_rate = .0001  # Initial learning rate.
	lr_update = 10  # Number of epochs to update the learning rate.
	val_step = 1000  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.

	# ## image encoder setting
	img_dim = 2048  # Dimensionality of the image embedding.
	# ## ### only for a bert image encoder
	trans_cfg = './itr/trans_cfg.json'  # config file for image transformer

	# ## text encoder setting
	text_encoder = 'bert'  # The RNN used for text encoder. lstm|gru|bert
	word_dim = 300  # Dimensionality of the word embedding.
	# ## ### only for a bert text encoder
	max_words = 32  # maximum number of words in a sentence.
	txt_stru = 'cnn'  # Whether to use pooling or cnn or rnn behind bert. 'pooling'/'cnn'/'rnn'/'trans'

	# ## fusion module setting and final joint embedding size
	embed_size = 1024  # Dimensionality of the joint embedding.
	final_dims = 256  # The dimension of followed extra encoder codes. Some methods use an extra encoder follow text/image encoder.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.


@ex.named_config
def SGRAF():  # AAAI 2021
	'''
	"Stacked Cross Attention for Image-Text Matching" is published on ECCV 2018. ("https://drive.google.com/file/d/1tAE_qkAxiw1CajjHix9EXoI7xu2t66iQ/view?usp=sharing"),
	The code original url is "https://github.com/Paranioar/SGRAF".
	'''

	name = "SGRAF"
	module_name = 'SAF'  # 'SGR', 'SAF'
	sgr_step = 3  # Step of the SGR.

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.
	num_epochs = 40  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0002  # Initial learning rate.
	lr_update = 30  # Number of epochs to update the learning rate.
	val_step = 1000  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.

	# ## image encoder setting
	img_dim = 2048  # Dimensionality of the image embedding.
	no_imgnorm = False  # Do not normalize the image embeddings.

	# ## text encoder setting
	bi_gru = True  # Use bidirectional GRU. When 'text_encoder' is 'bert', 'bi_gru' is a param for bert decoder feature.
	word_dim = 300  # Dimensionality of the word embedding.
	no_txtnorm = False  # Do not normalize the text embeddings.
	num_layers = 1  # Number of LSTM/GRU layers.

	# ## fusion module setting and final joint embedding size
	embed_size = 1024  # Dimensionality of the joint embedding.
	sim_dim = 256  # Dimensionality of the sim embedding.

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.


@ex.named_config
def CAMERA():  # ACM MM 2022
	'''
	"Context-Aware Multi-View Summarization Network for Image-Text Matching" is published on ACMM MM 2022. ("https://doi.org/10.1145/3394171.3413961"),
	The code original url is "https://github.com/LgQu/CAMERA".
	'''
	name = "CAMERA"

	# ## train setting
	resume = None  # path to latest checkpoint. If it is not None, load latest checkpoint from the path and train continue.

	num_epochs = 1  # Number of training epochs.
	batch_size = 128  # Size of a training mini-batch.
	learning_rate = .0001  # Initial learning rate.
	lr_update = 10  # Number of epochs to update the learning rate.
	val_step = 500  # Number of steps to run validation.
	log_step = 10  # Number of steps to print and recorrd the log.
	grad_clip = 2.  # Gradient clipping threshold.
	use_restval = False  # Use the restval data for training on MSCOCO.

	# ## image encoder setting
	img_dim = 2048  # Dimensionality of the image embedding.
	use_bbox = True  # Use region feature
	# ## ### CAMERA, AGSA is a module in text/image encoder.
	head = 64  # Number of heads in AGSA

	# ## text encoder setting
	text_encoder = 'bert'  # The RNN used for text encoder. lstm|gru|bert
	# ## ### only for a bert text encoder
	bert_path = '/workspace/dataset/uncased_L-12_H-768_A-12'  # The BERT model path.
	max_words = 32  # maximum number of words in a sentence.

	# ## fusion module setting and final joint embedding size
	embed_size = 2048  # Dimensionality of the joint embedding.
	measure = 'cosine'  # Similarity measure used (cosine|order)

	# ## loss setting
	max_violation = False  # Use max instead of sum in the rank loss.
	margin = 0.2  # Rank loss margin.
	# ## ### CAMERA
	smry_k = 12  # Number of views in summarization module
	smry_lamda = 0.01  # Trade-off in summarization module

	# CAMERA
	lr_decay_gamma = 0.1  # Decay coefficient for learning rate updating.
	drop = 0.0  # Dropout


@ex.config_hook
def hook(config, command_name, logger):
	if config['seed'] is None:
		config['seed'] = random.randint(0, 10000)
	config['name'] = 'VSE_PP' if config['name'] == 'VSE++' else config['name']
	if config['name'] not in __all__:
		raise ValueError(f'Please choose a method in {__all__}. {config["name"]} is given.')

	# make dir to save result
	save_dir = os.path.join(config['save_path'], config['name'],
							'_'.join([config['data_name'].split('_')[0], str(config['seed']),
									  time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())]))

	if not os.path.exists(config['save_path']):
		os.mkdir(config['save_path'])
	if not os.path.exists(os.path.join(config['save_path'], config['name'])):
		os.mkdir(os.path.join(config['save_path'], config['name']))
	if not os.path.exists(save_dir):
		os.mkdir(save_dir + config['tail']) if config['tail'] is not None else os.mkdir(save_dir)

	config["save_dir"] = save_dir

	if config['text_encoder'] == 'bert':
		config['vocab_file'] = os.path.join(config['bert_path'], 'vocab.txt')
		config['bert_config_file'] = os.path.join(config['bert_path'], 'bert_config.json')
		config['init_checkpoint'] = os.path.join(config['bert_path'], 'pytorch_model.bin')

	with open(os.path.join(save_dir, 'hparams.yaml'), 'w') as yaml_file:
		yaml.dump(config, yaml_file)

	# set cuda id
	os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda']

	return config
