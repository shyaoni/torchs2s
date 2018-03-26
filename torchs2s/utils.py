import numpy as np

import torch
from collections import Sequence

def list_sub(lst, removed):
    remain, subbed = [], []
    for x in lst:
        if x not in removed:
            remain.append(x)
        else:
            subbed.append(x)
    return remain, subbed

if __name__ == '__main__':
    pass
