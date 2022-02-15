# Introduction
This is the source code of Learning Fragment Self-Atention Embeddings for Image-Text Matching, ACM MM 2019.

## Requirements
* python 3.6
* pytorch 0.4.1

## Download data
We use the precomputed image features provided by [SCAN](https://github.com/kuanghuei/SCAN). Please download data.zip from [SCAN](https://github.com/kuanghuei/SCAN).

## Bert model
We use the bert code from [BERT-pytorch](https://github.com/huggingface/pytorch-transformers). Please following [here](https://github.com/huggingface/pytorch-transformers/blob/4fc9f9ef54e2ab250042c55b55a2e3c097858cb7/docs/source/converting_tensorflow_models.rst) to convert the Google bert model to a PyTorch save file.

## Training
```bash
python train.py --data_path /path/to/data --data_name f30k_precomp --bert_path /path/to/uncased_L-12_H-768_A-12/
python train.py --data_path /path/to/data --data_name coco_precomp --bert_path /path/to/uncased_L-12_H-768_A-12/
```
