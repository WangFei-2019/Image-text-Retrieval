import os
import json as jsonmod
import pickle

import nltk
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image

from . import tokenization
from . import vocab


def convert_to_feature(raw, seq_length, tokenizer):
	line = tokenization.convert_to_unicode(raw)
	tokens_a = tokenizer.tokenize(line)
	# Modifies `tokens_a` in place so that the total
	# length is less than the specified length.
	# Account for [CLS] and [SEP] with "- 2"
	if len(tokens_a) > seq_length - 2:
		tokens_a = tokens_a[0:(seq_length - 2)]

	tokens = list(tokens_a)
	tokens.insert(0, "[CLS]")
	tokens.insert(-1, "[SEP]")
	input_type_ids = [0] * len(tokens)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	if len(input_ids) < seq_length:
		fill_len = seq_length - len(input_ids)
		input_ids.extend([0] * fill_len)
		input_mask.extend([0] * fill_len)
		input_type_ids.extend([0] * fill_len)

	assert len(input_ids) == seq_length
	assert len(input_mask) == seq_length
	assert len(input_type_ids) == seq_length

	return tokens, input_ids, input_mask, input_type_ids


class PrecompDataset(data.Dataset):
	"""
	Load precomputed captions and image features
	Possible configions: f8k, f30k, coco, 10crop
	"""

	def __init__(self, data_path, data_split, config):
		self.config = config

		# Captions
		self.captions = []
		with open(os.path.join(data_path, '%s_caps.txt' % data_split), 'rb') as f:
			for line in tqdm.tqdm(f, desc=f'Loading {data_split} captions'):
				self.captions.append(line.strip())

		# Image features
		self.images = np.load(os.path.join(data_path, '%s_ims.npy' % data_split))
		if config['use_bbox']:
			self.boxes = np.load(os.path.join(data_path, '%s_boxes.npy' % data_split))
			self.img_wh = np.load(os.path.join(data_path, '%s_img_sizes.npy' % data_split))  # (num_img, 2)
		self.length = len(self.captions)
		# rkiros data has redundancy in images, we divide by 5, 10crop doesn't
		if self.images.shape[0] != self.length:
			self.im_div = 5
		else:
			self.im_div = 1
		# the development set for coco is large and so validation would be slow
		if data_split == 'dev':
			self.length = 5000

		# for BERT
		if self.config['text_encoder'] == 'bert':
			self.max_words = config['max_words']
			self.tokenizer = tokenization.FullTokenizer(
				vocab_file=self.config['vocab_file'], do_lower_case=True)
		else:
			self.vocab = vocab.deserialize_vocab(
				os.path.join(self.config['vocab_path'], '%s_vocab.json' % self.config['data_name'])) \
				if self.config['vocab_type'] is 'json' \
				else pickle.load(open(os.path.join(config['vocab_path'], '%s_vocab.pkl' % config['data_name']), 'rb'))

	def __getitem__(self, index):
		# handle the image redundancy
		img_id = index // self.im_div
		image = torch.Tensor(self.images[img_id])
		caption = self.captions[index]
		# Convert caption (string) to word ids.

		if self.config['use_bbox']:
			boxes = torch.Tensor(self.boxes[img_id])
			img_wh = torch.Tensor(self.img_wh[img_id])
		else:
			boxes, img_wh = None, None
		# for Bert
		if self.config['text_encoder'] == 'bert':
			tokens, captions_ids, captions_mask, captions_type_ids = convert_to_feature(caption, self.max_words,
																						self.tokenizer)
			captions_ids = torch.Tensor(captions_ids).long()
			captions_mask = torch.Tensor(captions_mask).long()
			captions_type_ids = torch.Tensor(captions_type_ids).long()
		else:
			tokens = nltk.tokenize.word_tokenize(str(caption).lower())
			tokens.insert(0, '<start>'), tokens.append('<end>')
			captions_ids = [self.vocab(token) for token in tokens]
			captions_ids = torch.Tensor(captions_ids).long()
			if self.config['name'] == 'VSRN':
				if len(captions_ids) > self.config['max_len']:
					captions_ids[self.config['max_len']] = captions_ids[-1]
					captions_ids = captions_ids[:self.config['max_len']]
				captions_ids = torch.cat((captions_ids, torch.zeros(self.config['max_len'] + 1 - len(captions_ids))), 0)

				captions_mask = torch.zeros(self.config['max_len'] + 1)
				captions_mask[:min(len(captions_ids), self.config['max_len'])] = 1
				captions_type_ids = None
			else:
				captions_mask, captions_type_ids = None, None
		return image, boxes, img_wh, captions_ids, index, img_id, captions_mask, captions_type_ids

	def __len__(self):
		return self.length


def collate_fn(data):
	"""Build mini-batch tensors from a list of (image, caption) tuples.
	Args:
		data: list of (image, caption) tuple.
			- image: torch tensor of shape (3, 256, 256).
			- caption: torch tensor of shape (?); variable length.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length
	data.sort(key=lambda x: len(x[3]), reverse=True)

	images, boxes, imgs_wh, captions_ids, ids, img_ids, captions_mask, captions_type_ids = zip(*data)

	if not None in boxes:
		# Merge images (convert tuple of 3D tensor to 4D tensor)
		images = torch.stack(images, 0)
		boxes = torch.stack(boxes, 0)
		imgs_wh = torch.stack(imgs_wh, 0)

		# Merget captions (convert tuple of 1D tensor to 2D tensor)
		lengths = [torch.sum(cap_m) for cap_m in captions_mask]
		captions_ids = torch.stack(captions_ids, 0)
		captions_mask = torch.stack(captions_mask, 0)
		captions_type_ids = torch.stack(captions_type_ids, 0)

		ids = np.array(ids)
	else:
		# Merge images (convert tuple of 3D tensor to 4D tensor)
		images = torch.stack(images, 0)

		# Merget captions (convert tuple of 1D tensor to 2D tensor)
		lengths = [len(cap) for cap in captions_ids]
		targets = torch.zeros(len(captions_ids), max(lengths)).long()
		for i, cap in enumerate(captions_ids):
			end = lengths[i]
			targets[i, :end] = cap[:end]
		captions_ids = targets
		if not None in captions_mask:
			captions_mask = torch.stack(captions_mask, 0)
		if not None in captions_type_ids:
			captions_type_ids = torch.stack(captions_type_ids, 0)
	return images, boxes, imgs_wh, captions_ids, lengths, ids, captions_mask, captions_type_ids


def get_precomp_loader(data_path, data_split, config, batch_size=100,
					   shuffle=True, num_workers=5):
	"""Returns torch.utils.data.DataLoader for custom coco dataset."""
	dset = PrecompDataset(data_path, data_split, config)
	if config['text_encoder'] != 'bert':
		vocab_size = len(dset.vocab)
	else:
		vocab_size = len(dset.tokenizer.vocab)
	data_loader = torch.utils.data.DataLoader(dataset=dset,
											  batch_size=batch_size,
											  shuffle=shuffle,
											  pin_memory=True,
											  collate_fn=collate_fn,
											  num_workers=num_workers,
											  )
	return data_loader, vocab_size


def get_loaders(data_name, batch_size, workers, config):
	dpath = os.path.join(config['data_path'], data_name)
	if config['data_name'].endswith('_precomp'):
		train_loader, vocab_size = get_precomp_loader(dpath, 'train', config, batch_size, True, workers)
		val_loader, vocab_size = get_precomp_loader(dpath, 'dev', config, batch_size, False, workers)
	else:
		vocabs = pickle.load(open(os.path.join(config['vocab_path'], '%s_vocab.pkl' % config['data_name']), 'rb'))
		# Build Dataset Loader
		roots, ids = get_paths(dpath, data_name, config['use_restval'])

		transform = get_transform(data_name, 'train', config)
		train_loader = get_loader_single(config['data_name'], 'train',
										 roots['train']['img'],
										 roots['train']['cap'],
										 vocabs, transform, ids=ids['train'],
										 batch_size=batch_size, shuffle=True,
										 num_workers=workers,
										 collate_fn=collate_fn)

		transform = get_transform(data_name, 'val', config)
		val_loader = get_loader_single(config['data_name'], 'val',
									   roots['val']['img'],
									   roots['val']['cap'],
									   vocabs, transform, ids=ids['val'],
									   batch_size=batch_size, shuffle=False,
									   num_workers=workers,
									   collate_fn=collate_fn)
		vocab_size = len(vocabs)

	return train_loader, val_loader, vocab_size


def get_test_loader(split_name, data_name, batch_size, workers, config):
	dpath = os.path.join(config['data_path'], data_name)
	test_loader, vocab_size = get_precomp_loader(dpath, split_name, config, batch_size, False, workers)

	return test_loader, vocab_size


# VSE++
def get_paths(path, name='coco', use_restval=False):
	"""
	Returns paths to images and annotations for the given datasets. For MSCOCO
	indices are also returned to control the data split being used.
	The indices are extracted from the Karpathy et al. splits using this
	snippet:

	# >>> import json
	# >>> dataset=json.load(open('dataset_coco.json','r'))
	# >>> A=[]
	# >>> for i in range(len(D['images'])):
	# ...   if D['images'][i]['split'] == 'val':
	# ...     A+=D['images'][i]['sentids'][:5]
	# ...

	:param name: Dataset names
	:param use_restval: If True, the the `restval` data is included in train.
	"""
	roots = {}
	ids = {}
	if 'coco' == name:
		imgdir = os.path.join(path, 'images')
		capdir = os.path.join(path, 'annotations')
		roots['train'] = {
			'img': os.path.join(imgdir, 'train2014'),
			'cap': os.path.join(capdir, 'captions_train2014.json')
		}
		roots['val'] = {
			'img': os.path.join(imgdir, 'val2014'),
			'cap': os.path.join(capdir, 'captions_val2014.json')
		}
		roots['test'] = {
			'img': os.path.join(imgdir, 'val2014'),
			'cap': os.path.join(capdir, 'captions_val2014.json')
		}
		roots['trainrestval'] = {
			'img': (roots['train']['img'], roots['val']['img']),
			'cap': (roots['train']['cap'], roots['val']['cap'])
		}
		ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
		ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
		ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
		ids['trainrestval'] = (
			ids['train'],
			np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
		if use_restval:
			roots['train'] = roots['trainrestval']
			ids['train'] = ids['trainrestval']
	elif 'f8k' == name:
		imgdir = os.path.join(path, 'images')
		cap = os.path.join(path, 'dataset_flickr8k.json')
		roots['train'] = {'img': imgdir, 'cap': cap}
		roots['val'] = {'img': imgdir, 'cap': cap}
		roots['test'] = {'img': imgdir, 'cap': cap}
		ids = {'train': None, 'val': None, 'test': None}
	elif 'f30k' == name:
		imgdir = os.path.join(path, 'images')
		cap = os.path.join(path, 'dataset_flickr30k.json')
		roots['train'] = {'img': imgdir, 'cap': cap}
		roots['val'] = {'img': imgdir, 'cap': cap}
		roots['test'] = {'img': imgdir, 'cap': cap}
		ids = {'train': None, 'val': None, 'test': None}

	return roots, ids


def get_transform(data_name, split_name, config):
	normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									  std=[0.229, 0.224, 0.225])
	t_list = []
	if split_name == 'train':
		t_list = [transforms.RandomResizedCrop(config['crop_size']),
				  transforms.RandomHorizontalFlip()]
	elif split_name == 'val':
		t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
	elif split_name == 'test':
		t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

	t_end = [transforms.ToTensor(), normalizer]
	transform = transforms.Compose(t_list + t_end)
	return transform


def get_loader_single(data_name, split, root, json, vocabs, transform,
					  batch_size=100, shuffle=True,
					  num_workers=2, ids=None, collate_fn=collate_fn):
	"""Returns torch.utils.data.DataLoader for custom coco dataset."""
	if 'coco' in data_name:
		# COCO custom dataset
		dataset = CocoDataset(root=root,
							  json=json,
							  vocabs=vocabs,
							  transform=transform, ids=ids)
	elif 'f8k' in data_name or 'f30k' in data_name:
		dataset = FlickrDataset(root=root,
								split=split,
								json=json,
								vocabs=vocabs,
								transform=transform)

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
											  batch_size=batch_size,
											  shuffle=shuffle,
											  pin_memory=True,
											  num_workers=num_workers,
											  collate_fn=collate_fn)
	return data_loader


class CocoDataset(data.Dataset):
	"""COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

	def __init__(self, root, json, vocab, transform=None, ids=None):
		"""
		Args:
			root: image directory.
			json: coco annotation file path.
			vocab: vocabulary wrapper.
			transform: transformer for image.
		"""
		self.root = root
		# when using `restval`, two json files are needed
		if isinstance(json, tuple):
			self.coco = (COCO(json[0]), COCO(json[1]))
		else:
			self.coco = (COCO(json),)
			self.root = (root,)
		# if ids provided by get_paths, use split-specific ids
		if ids is None:
			self.ids = list(self.coco.anns.keys())
		else:
			self.ids = ids

		# if `restval` data is to be used, record the break point for ids
		if isinstance(self.ids, tuple):
			self.bp = len(self.ids[0])
			self.ids = list(self.ids[0]) + list(self.ids[1])
		else:
			self.bp = len(self.ids)
		self.vocab = vocab
		self.transform = transform

	def __getitem__(self, index):
		"""This function returns a tuple that is further passed to collate_fn
		"""
		vocab = self.vocab
		root, caption, img_id, path, image = self.get_raw_item(index)

		if self.transform is not None:
			image = self.transform(image)

		# Convert caption (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(
			str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		return image, target, index, img_id

	def get_raw_item(self, index):
		if index < self.bp:
			coco = self.coco[0]
			root = self.root[0]
		else:
			coco = self.coco[1]
			root = self.root[1]
		ann_id = self.ids[index]
		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']
		image = Image.open(os.path.join(root, path)).convert('RGB')

		return root, caption, img_id, path, image

	def __len__(self):
		return len(self.ids)


class FlickrDataset(data.Dataset):
	"""
	Dataset loader for Flickr30k and Flickr8k full datasets.
	"""

	def __init__(self, root, json, split, vocabs, transform=None):
		self.root = root
		self.vocab = vocabs
		self.split = split
		self.transform = transform
		self.dataset = jsonmod.load(open(json, 'r'))['images']
		self.ids = []
		for i, d in enumerate(self.dataset):
			if d['split'] == split:
				self.ids += [(i, x) for x in range(len(d['sentences']))]

	def __getitem__(self, index):
		"""This function returns a tuple that is further passed to collate_fn
		"""
		vocabs = self.vocab
		root = self.root
		ann_id = self.ids[index]
		img_id = ann_id[0]
		caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
		path = self.dataset[img_id]['filename']

		image = Image.open(os.path.join(root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		# Convert caption (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(
			str(caption).lower())
		caption = []
		caption.append(vocabs('<start>'))
		caption.extend([vocabs(token) for token in tokens])
		caption.append(vocabs('<end>'))
		target = torch.Tensor(caption)
		return image, target, index, img_id

	def __len__(self):
		return len(self.ids)
