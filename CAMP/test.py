from vocab import Vocabulary
import evaluation
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
evaluation.evalrank("/workspace/Image-Text_Retrieval/CAMP/runs/f30k_cros_attn_256_new_normalLe_0.002_seed100_20220128-01-34-32-withH/model_best.pth.tar", data_path="/workspace/Image-Text_Retrieval/datasets", split="test",
                     fold5=False)
# 这个test没有用，看好readme
"""print (rt,rti)
print(len(rt),len(rti))
dic_now = {}
dic_now["rt_ranks"]=rt[0]
dic_now["rt_top1"]=rt[1]
dic_now["rti_ranks"] =rti[0]
dic_now["rti_top1"]=rti[1]

with open('vsepp' + '.results.pickle', 'wb') as handle:
    pickle.dump(dic_now, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
