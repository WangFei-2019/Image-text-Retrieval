# Where Does the Performance Improvement Come From? \- A Reproducibility Concern about Image-Text Retrieval
PyTorch code of the paper "Where Does the Performance Improvement Come From? - A Reproducibility Concern about Image-Text Retrieval". It includes five models of [VSE++](https://github.com/fartashf/vsepp), [SCAN](https://github.com/kuanghuei/SCAN), [VSRN](https://github.com/KunpengLi1994/VSRN), [SAEM](https://github.com/yiling2018/saem), [SGRAF](https://github.com/Paranioar/SGRAF) and [CAMERA](https://github.com/LgQu/CAMERA).

Jun Rao, Fei Wang, Liang Ding, Shuhan. Qi, Yibin Zhan, Weifeng Liu, and Dacheng Tao, [“Where does the performance improvement come from - a reproducibility concern about image-text retrieval,”](https://arxiv.org/abs/2203.03853) in SIGIR, 2022.

# Contents
1. [Introduction](#introduction)
2. [Code](#code)

    * [Requirement](#requirement)
    * [Down data and vocab](#down-data-and-vocab)
    * [Pretrained BERT model](#pretrained-bert-model)
    * [Train](#train)
        
        * [VSE++](#vse)
        * [SCAN](#scan)
        * [VSRN](#vsrn)
        * [SAEM](#saem)
        * [SGRAF](#sgraf)
        * [CAMERA](#camera)
    * [Test](#test)
3. [Statement](#statement)
4. [Vision](#vision)
5. [License](#license)
6. [Citation](#citation)


# Introduction
This article aims to provide the information retrieval community with some reflections on recent advances in retrieval learning by analyzing the reproducibility of image-text retrieval models. Due to the increase of multimodal data over the last decade, image-text retrieval has steadily become a major research direction in the field of information retrieval. Numerous researchers train and evaluate image-text retrieval algorithms using benchmark datasets such as MS-COCO and Flickr30k. Research in the past has mostly focused on performance, with multiple state-of-the-art methodologies being suggested in a variety of ways. According to their assertions, these techniques provide improved modality interactions and hence more precise multimodal representations. In contrast to previous works, we focus on the reproducibility of the approaches and the examination of the elements that lead to improved performance by pretrained and nonpretrained models in retrieving images and text.

To be more specific, we first examine the related reproducibility concerns and explain why our focus is on image-text retrieval tasks. Second, we systematically summarize the current paradigm of image-text retrieval models and the stated contributions of those approaches. Third, we analyze various aspects of the reproduction of pretrained and nonpretrained retrieval models. To complete this, we conducted ablation experiments and obtained some influencing factors that affect retrieval recall more than the improvement claimed in the original paper. Finally, we present some reflections and challenges that the retrieval community should consider in the future. Our source code is publicly available at https://github.com/WangFei-2019/Image-text-Retrieval.

![model](./fig/framework.png)

# Code
We change all sub-project code to fit torch1.7 and CUDA11 and add random seed for all methods. You can use it from [code](https://github.com/WangFei-2019/Image-text-Retrieval).

## Requirement
We recommended the following dependencies.

* Python 3.7  
* [PyTorch](http://pytorch.org/) (1.7.1)
* [NumPy](http://www.numpy.org/) (1.19.5)
* [torchvision]()(0.8.2)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```
## Down data and vocab
We follow [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention) and [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features for fair comparison. 
More details about data pre-processing (optional) can be found [here](https://github.com/kuanghuei/SCAN/blob/master/README.md#data-pre-processing-optional). 
All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
# You can also get the data from google drive: https://drive.google.com/drive/u/1/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC.
```

We use [bottom-up attention](https://github.com/peteanderson80/bottom-up-attention) to extract the positions of detected boxes, including coordinate,  width and height,  which can be downloaded from https://drive.google.com/file/d/1K9LnWJc71dK6lF1BJMPlbkIu_vYmHjVP/view?usp=sharing. You can put MSCOCO/Flickr30K data in a same file.

We refer to the path of extracted files as `$DATA_PATH`. 

[An example](#datapath) for a `$DATA_PATH`.


## Pretrained BERT model
We use the BERT code from [BERT-pytorch](https://github.com/huggingface/pytorch-transformers). Please following [here](https://github.com/huggingface/pytorch-transformers/blob/4fc9f9ef54e2ab250042c55b55a2e3c097858cb7/docs/source/converting_tensorflow_models.rst) to convert the Google BERT model to a PyTorch save file `$BERT_PATH`.

## Train
You can see the details of all hyperparams in [config.py](./itr/config.py). 
If you want to know more detail about each method, look at the README of each method project in barch "original". 

```bash
# An example for training.
python train.py with "$METHOD_NAME" data_path="$DATA_PATH" data_name="$DATA_NAME"
```

### VSE++
```bash
python train.py with VSE_PP data_path="$DATA_PATH" data_name="$DATA_NAME" max_violation=True
```

### SCAN
#### t-i LSE 
```bash
# For MSCOCO
python train.py with SCAN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True max_violation=True bi_gru=True agg_func=LogSumExp cross_attn=t2i lambda_lse=6 lambda_softmax=9
#For Flickr30K
python train.py with SCAN data_path="$DATA_PATH" data_name=f30k_precomp max_violation=True max_violation=True bi_gru=True agg_func=LogSumExp cross_attn=t2i lambda_lse=6 lambda_softmax=9
```
#### t-i AVG 
```bash
# For MSCOCO
python train.py with SCAN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True max_violation=True bi_gru=True agg_func=Mean cross_attn=t2i lambda_lse=6 lambda_softmax=9
# For Flickr30K
python train.py with SCAN data_path="$DATA_PATH" data_name=f30k_precomp max_violation=True max_violation=True bi_gru=True agg_func=Mean cross_attn=t2i lambda_lse=6 lambda_softmax=9
```
#### i-t LSE 
```bash
# For MSCOCO
python train.py with SCAN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True max_violation=True bi_gru=True agg_func=LogSumExp cross_attn=i2t lambda_lse=20 lambda_softmax=4
# For Flickr30K
python train.py with SCAN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True max_violation=True bi_gru=True agg_func=LogSumExp cross_attn=i2t lambda_lse=5 lambda_softmax=4
```
#### i-t AVG 
```bash
# For MSCOCO
python train.py with SCAN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True max_violation=True bi_gru=True agg_func=Mean cross_attn=i2t lambda_lse=6 lambda_softmax=4
# For Flickr30K
python train.py with SCAN data_path="$DATA_PATH" data_name=f30k_precomp max_violation=True max_violation=True bi_gru=True agg_func=Mean cross_attn=i2t lambda_lse=6 lambda_softmax=4
```

### VSRN
```bash
# For MSCOCO
python train.py with VSRN data_path="$DATA_PATH" data_name=coco_precomp max_violation=True lr_update=15
# For Flickr30K
python train.py with VSRN data_path="$DATA_PATH" data_name=f30k_precomp max_violation=True lr_update=10
```

### SAEM
```bash
python train.py with SAEM data_path="$DATA_PATH" data_name="$DATA_NAME" max_violation=True bert_path="$BERT_PATH"
```

### SGRAF
#### SGR
```bash
# For MSCOCO
python train.py with SGRAF data_path="$DATA_PATH" data_name=coco_precomp module_name=SGR max_violation=True num_epochs=20 lr_update=10
# For Flickr30K
python train.py with SGRAF data_path="$DATA_PATH" data_name=f30k_precomp module_name=SGR max_violation=True num_epochs=40 lr_update=30
```
#### SAF
```bash
# For MSCOCO
python train.py with SGRAF data_path="$DATA_PATH" data_name=coco_precomp module_name=SAF max_violation=True num_epochs=20 lr_update=10
# For Flickr30K
python train.py with SGRAF data_path="$DATA_PATH" data_name=f30k_precomp module_name=SAF max_violation=True num_epochs=30 lr_update=20
```

### CAMERA
```bash
# For MSCOCO
python train.py with CAMERA data_path="$DATA_PATH" data_name=coco_precomp bert_path="$BERT_PATH" max_violation=True num_epochs=40 lr_update=20
# For Flickr30K
python train.py with CAMERA data_path="$DATA_PATH" data_name=coco_precomp bert_path="$BERT_PATH" max_violation=True num_epochs=30 lr_update=10
```

## Test 
There is a complete test progress in [test.py](./test.py).
```python
from itr.metricmodule import evaluation

# Evaluate A Single Modal.
DATA_PATH = None  # If test data path is different from train data path, please give a new path to test.
MODEL_PATH = '$MODEL_PATH'
# ## Test on Flickr30k
evaluation.evalrank_single(model_path=MODEL_PATH, data_path=DATA_PATH, split='test')
# ## Test on MSCOCO (1000test→fold5=True; 5000test→fold5=False)
evaluation.evalrank_single(model_path=MODEL_PATH, data_path=DATA_PATH, split='testall', fold5=True)


# Evaluate The Ensemble Modal.
DATA_PATH = None  # If test data path is different from train data path, please give a new path to test.
MODEL_PATH_1 = '$MODEL_PATH'
MODEL_PATH_2 = '$MODEL_PATH'
# ## Test on Flickr30k
evaluation.evalrank_ensemble(model_path=MODEL_PATH_1, model_path2=MODEL_PATH_2, data_path=DATA_PATH, split='test')
## Test on MSCOCO (1000test→fold5=True; 5000test→fold5=False)
evaluation.evalrank_ensemble(model_path=MODEL_PATH_1, model_path2=MODEL_PATH_2, data_path=DATA_PATH, split='testall', fold5=False)
```
# Statement
In the research, we found that the image-text retrieval community need a unified project for a funture research. We publish a beta vision for Image-Test retrieval research. Every researcher is welcome to test the code, we will accept your valuable comments and reply in the issue area.

# Vision
bate-v0.1

# License
The license is CC-BY-NC 4.0.

# Citation

### Please cite as:

```bibtex
@inproceedings{rao2022reproducibility,
    title = {Where Does the Performance Improvement Come From - A Reproducibility Concern about Image-Text Retrieval},
    author = {Jun Rao and Fei Wang and Liang Ding and Shuhan Qi and Yibing Zhan and Weifeng Liu and Dacheng Tao},
    booktitle = {SIGIR},
    year = {2022}
}
```

#### DATA_PATH
```
+-- $DATA_PATH
    +-- coco_precomp
        |-- dev_boxes.npy
        |-- dev_caps.txt
        |-- dev_ids.txt
        |-- dev_img_sizes.npy
        |-- dev_ims.npy
        |-- testall_boxes.npy
        |-- testall_caps.txt
        |-- testall_ids.txt
        |-- testall_img_sizes.npy
        |-- testall_ims.npy
        |-- test_boxes.npy
        |-- test_caps.txt
        |-- test_ids.txt
        |-- test_img_sizes.npy
        |-- test_ims.npy
        |-- train_boxes.npy
        |-- train_caps.txt
        |-- train_ids.txt
        |-- train_img_sizes.npy
        |-- train_ims.npy
    +-- f30k_precomp
        |-- dev_boxes.npy
        |-- dev_caps.txt
        |-- dev_ids.txt
        |-- dev_img_sizes.npy
        |-- dev_ims.npy
        |-- dev_tags.txt
        |-- test_boxes.npy
        |-- test_caps.txt
        |-- test_ids.txt
        |-- test_img_sizes.npy
        |-- test_ims.npy
        |-- test_tags.txt
        |-- train_boxes.npy
        |-- train_caps.txt
        |-- train_ids.txt
        |-- train_img_sizes.npy
        |-- train_ims.npy
        |-- train_tags.txt
    +-- coco
    +-- f30k
    +-- 10crop_precomp
```