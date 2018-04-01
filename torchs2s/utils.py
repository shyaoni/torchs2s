import torch
from torch.autograd import Variable

def is_tensor(tensor):
    """check whether tensor is a Variable, FloatTensor, LongTensor
    """
    return isinstance(tensor, Variable) or \
           isinstance(tensor, torch.FloatTensor) or \
           isinstance(tensor, torch.LongTensor) or \
           isinstance(tensor, torch.cuda.FloatTensor) or \
           isinstance(tensor, torch.cuda.LongTensor)

def is_sequence(x):
    """check whether object x is a list or tuple. """
    return isinstance(x, list) or isinstance(x, tuple)

def list_sub(lst, removed):
    """list_sub(a, b) -> a \ b, a \ (a \ b)"""
    remain, subbed = [], []
    for x in lst:
        if x not in removed:
            remain.append(x)
        else:
            subbed.append(x)
    return remain, subbed

def mask(lengths):
    """get the 0/1 mask tensor according to lengths where the mask will be 
    placed along the first dim."""
    shape = lengths.shape

    lengths = lengths.view(-1)
    p = lengths.max()
    
    tensor = torch.zeros(p, lengths.shape[0])
    for i, l in enumerate(lengths):
        tensor[:l, i] = 1

    return tensor.view((p,) + shape)
