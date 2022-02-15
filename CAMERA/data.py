import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import tqdm
from PIL import Image
# from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import random
import tokenization
import numpy as np
import time
import pickle
import tqdm


def convert_to_feature(raw, seq_length, tokenizer):
    line = tokenization.convert_to_unicode(raw)
    tokens_a = tokenizer.tokenize(line)
    # Modifies `tokens_a` in place so that the total
    # length is less than the specified length.
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return tokens, input_ids, input_mask, input_type_ids


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, opt):
        self.opt = opt
        loc = data_path + '/'

        # Captions
        self.captions = []
        token_caption = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in tqdm.tqdm(f):
                self.captions.append(line.strip())
                tokens = nltk.tokenize.word_tokenize(str(line.strip()).lower())
                # tokens = nltk.tokenize.word_tokenize(str(line.strip()).lower().decode('utf-8'))
                token_caption.append(tokens)

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.boxes = np.load(loc + '%s_boxes.npy' % data_split)
        self.img_wh = np.load(loc + '%s_img_sizes.npy' % data_split)  # (num_img, 2)

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.max_words = opt.max_words  # for BERT
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.from_numpy(self.images[img_id])
        box = torch.from_numpy(self.boxes[img_id])
        img_wh = torch.from_numpy(self.img_wh[img_id])
        caption = self.captions[index]

        # for Bert
        tokens, input_ids, input_mask, input_type_ids = convert_to_feature(caption, self.max_words, self.tokenizer)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)

        return image, box, img_wh, input_ids, index, img_id, input_mask, input_type_ids

    def __len__(self):
        return self.length


# b_ids = torch.randperm(36)

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
    data.sort(key=lambda x: torch.sum(x[-2]), reverse=True)
    images, boxes, imgs_wh, input_ids, ids, img_ids, input_mask, input_type_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    boxes = torch.stack(boxes, 0)
    imgs_wh = torch.stack(imgs_wh, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [torch.sum(cap) for cap in input_mask]
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    input_type_ids = torch.stack(input_type_ids, 0)

    ids = np.array(ids)

    return images, boxes, imgs_wh, input_ids, lengths, ids, input_mask, input_type_ids


def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=5):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', opt,
                                    batch_size, False, workers)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt, \
                                     batch_size, False, workers)

    return test_loader
