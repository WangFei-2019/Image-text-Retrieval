import torch
import torch.nn as nn
from torch.autograd import Variable
import os 
import random



def print_options(args):
    print("")
    print("----- options -----".center(120, '-'))
    args = vars(args)
    string = ''
    for i, (k, v) in enumerate(sorted(args.items())):
        string += "{}: {}".format(k, v).center(40, ' ')
        if i % 3 == 2 or i == len(args.items()) - 1:
            print(string)
            string = ''
    print("".center(120, '-'))
    print("")


