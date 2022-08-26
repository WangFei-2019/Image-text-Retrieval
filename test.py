from itr.metricmodule import evaluation

# # Evaluate A Single Modal.
# DATA_PATH = None  # If test data path is different from train data path, please give a new path to test.
# MODEL_PATH = 'MODEL_PATH'
# # ## Test on Flickr30k
# evaluation.evalrank_single(model_path=MODEL_PATH, data_path=DATA_PATH, split='test')
# # ## Test on MSCOCO (1000test→fold5=True; 5000test→fold5=False)
# evaluation.evalrank_single(model_path=MODEL_PATH, data_path=DATA_PATH, split='testall', fold5=True)


# Evaluate The Ensemble Modal.
DATA_PATH = None  # If test data path is different from train data path, please give a new path to test.
MODEL_PATH_1 = '/workspace/ITR/runs/CAMERA/coco_0_2022-08-26-03-13-09/model_best.pth.tar'
MODEL_PATH_2 = '/workspace/ITR/runs/CAMERA/coco_0_2022-08-26-05-23-48/model_best.pth.tar'
# # ## Test on Flickr30k
# evaluation.evalrank_ensemble(model_path=MODEL_PATH_1, model_path2=MODEL_PATH_2, data_path=DATA_PATH, split='test')
# ## Test on MSCOCO (1000test→fold5=True; 5000test→fold5=False)
evaluation.evalrank_ensemble(model_path=MODEL_PATH_1, model_path2=MODEL_PATH_2, data_path=DATA_PATH, split='testall', fold5=False)

