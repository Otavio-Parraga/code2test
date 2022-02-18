import os
import numpy as np
import torch
import random

def tuple_to_device(tup, device):
    return tuple(d.to(device) for d in tup)


def dict_to_device(dict, device):
    return {k: v.to(device) for k, v in dict.items()}


def squeeze_dict(dict):
    return {k: v.squeeze(0) for k, v in dict.items()}

def set_seed(seed=42):

    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True