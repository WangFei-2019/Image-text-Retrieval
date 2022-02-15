import torch
import torch.nn as nn
import torch.nn.functional as F

class Summarization(nn.Module):
    ''' Multi-View Summarization Module '''
    def __init__(self, embed_size, smry_k):
        super(Summarization, self).__init__()
        # dilation conv
        out_c = [256, 128, 128, 128, 128, 128, 128]
        k_size = [1, 3, 3, 3, 5, 5, 5]
        dila = [1, 1, 2, 3, 1, 2, 3]
        pads = [0, 1, 2, 3, 2, 4, 6]
        convs_dilate = [nn.Conv1d(embed_size, out_c[i], k_size[i], dilation=dila[i], padding=pads[i]) \
                        for i in range(len(out_c))]
        self.convs_dilate = nn.ModuleList(convs_dilate)
        self.convs_fc = nn.Linear(1024, smry_k)

    def forward(self, rgn_emb):
        x = rgn_emb.transpose(1, 2)    #(bs, dim, num_r)
        x = [F.relu(conv(x)) for conv in self.convs_dilate]
        x = torch.cat(x, dim=1) #(bs, 1024, num_r)
        x = x.transpose(1, 2)   #(bs, num_r, 1024)
        smry_mat = self.convs_fc(x)    #(bs, num_r, k)
        return smry_mat