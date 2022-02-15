import torch
from vocab import Vocabulary
import evaluation_models

# for coco
# print('Evaluation on COCO:')
# evaluation_models.evalrank("/workspace/Image-Text_Retrieval/VSRN/runs/coco_VSRN_nh_seed100_20220209-18-45-00/model_best.pth.tar", "/workspace/Image-Text_Retrieval/VSRN/runs/coco_VSRN_nh_seed100_20220209-18-45-00/model_best.pth.tar", data_path='/workspace/Image-Text_Retrieval/datasets', split="testall", fold5=True)

# # for flickr

print('Evaluation on Flickr30K:')
evaluation_models.evalrank("/workspace/Image-Text_Retrieval/VSRN/runs10/flickr_VSRN_seed1000/model_best.pth.tar", "/workspace/Image-Text_Retrieval/VSRN/runs10/flickr_VSRN_seed1000/model_best.pth.tar", data_path='/workspace/Image-Text_Retrieval/datasets', split="test", fold5=False)
