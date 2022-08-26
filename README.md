# Where Does the Performance Improvement Come From? \- A Reproducibility Concern about Image-Text Retrieval
PyTorch code of the paper "Where Does the Performance Improvement Come From? - A Reproducibility Concern about Image-Text Retrieval". It includes five models of [VSE++](https://github.com/fartashf/vsepp), [SCAN](https://github.com/kuanghuei/SCAN), [VSRN](https://github.com/KunpengLi1994/VSRN), [SAEM](https://github.com/yiling2018/saem), [SGRAF](https://github.com/Paranioar/SGRAF) and [CAMERA](https://github.com/LgQu/CAMERA).

J. Rao, F. Wang, L. Ding, S. Qi, Y. Zhan, W. Liu, and D. Tao, [“Where does the performance improvement come from - a reproducibility concern about image-text retrieval,”](https://arxiv.org/abs/2203.03853) in SIGIR, 2022.

## Introduction
This article aims to provide the information retrieval community with some reflections on recent advances in retrieval learning by analyzing the reproducibility of image-text retrieval models. Due to the increase of multimodal data over the last decade, image-text retrieval has steadily become a major research direction in the field of information retrieval. Numerous researchers train and evaluate image-text retrieval algorithms using benchmark datasets such as MS-COCO and Flickr30k. Research in the past has mostly focused on performance, with multiple state-of-the-art methodologies being suggested in a variety of ways. According to their assertions, these techniques provide improved modality interactions and hence more precise multimodal representations. In contrast to previous works, we focus on the reproducibility of the approaches and the examination of the elements that lead to improved performance by pretrained and nonpretrained models in retrieving images and text.

To be more specific, we first examine the related reproducibility concerns and explain why our focus is on image-text retrieval tasks. Second, we systematically summarize the current paradigm of image-text retrieval models and the stated contributions of those approaches. Third, we analyze various aspects of the reproduction of pretrained and nonpretrained retrieval models. To complete this, we conducted ablation experiments and obtained some influencing factors that affect retrieval recall more than the improvement claimed in the original paper. Finally, we present some reflections and challenges that the retrieval community should consider in the future. Our source code is publicly available at https://github.com/WangFei-2019/Image-text-Retrieval.



# License
The license is CC-BY-NC 4.0.

# Citation

Please cite as:

```bibtex
@inproceedings{rao2022reproducibility,
    title = {Where Does the Performance Improvement Come From - A Reproducibility Concern about Image-Text Retrieval},
    author = {Jun Rao and Fei Wang and Liang Ding and Shuhan Qi and Yibing Zhan and Weifeng Liu and Dacheng Tao},
    booktitle = {SIGIR},
    year = {2022}
}
```
