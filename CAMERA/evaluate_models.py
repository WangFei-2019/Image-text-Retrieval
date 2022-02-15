import torch
import evaluation_models

DATA_PATH = '/workspace/Image-Text_Retrieval/datasets'
# flickr
# evaluation_models.evalrank_single("/workspace/Image-Text_Retrieval/CAMERA/runs/flickr_nh_seed100_20220205-17-33-53/model_best.pth.tar", data_path=DATA_PATH, split="test", fold5=False)
# # evaluation_models.evalrank_ensemble("pretrain_model/flickr/model_flickr_1.pth.tar", "pretrain_model/flickr/model_flickr_2.pth.tar", \
# #                     data_path=DATA_PATH, split="test", fold5=False)

# coco
evaluation_models.evalrank_single("/workspace/Image-Text_Retrieval/CAMERA/runs/coco_nh_new_seed100_20220212-16-43-32/model_best.pth.tar", data_path=DATA_PATH, split="testall", fold5=False)
# evaluation_models.evalrank_ensemble("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", \
#                     data_path=DATA_PATH, split="testall", fold5=True)










