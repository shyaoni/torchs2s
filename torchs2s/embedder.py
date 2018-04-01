from tqdm import tqdm
import torch

import torchs2s.utils as utils

from torch.nn import Embedding 

def get_init_func(dim, init_func=None, initializer=None):
    if initializer is None:
        initializer = lambda dim: torch.randn(dim)
    if init_func is None:
        init_func = lambda x, dim: torch.randn(dim)
    return init_func

def complete_with_init_func(dim, vocab, tensors, **kwargs):
    init_func = get_init_func(dim, **kwargs)
    if tensors is None:
        tensors = [None,] * vocab.size
    for i, tensor in enumerate(tensors):
        if tensor is None:
            tensors[i] = init_func(vocab.itos[i], dim)
    return torch.stack(tensors, dim=0)

def load_from_glove(dim, vocab, path, **kwargs):
    tensors = [None,] * vocab.size
    with open(path, 'r') as f:
        for line in f:
            vecs = line.strip().split(' ')
            word, vecs = vecs[0], vecs[1:]
            dim = len(vecs)

            if word in vocab.stoi:
                idx = vocab.stoi[word]
                tensors[idx] = torch.FloatTensor([float(v) for v in vecs])

    return complete_with_init_func(dim, vocab, tensors, **kwargs)

def load_from(dim, vocab, load=None, path=None, **kwargs):
    if load is None:
        load = 'glove'

    if isinstance(load, str):
        if path is None:
            return complete_with_init_func(dim, vocab, None, **kwargs) 
        if load == 'glove':
            return load_from_glove(dim, vocab, path, **kwargs) 

class Embedder(Embedding):
    def __init__(self, dim, vocab, 
                 load=None, path=None, init_func=None, initializer=None, 
                 **kwargs):
        tensor = load_from(dim, vocab, load, path, **kwargs)
        super().__init__(tensor.shape[0], tensor.shape[1], **kwargs)
        self.weight.data.copy_(tensor)

    def forward(self, x):
        if x.dim() == 2:
            return super().forward(x)
        else:
            shape = x.shape + (self.embedding_dim, )
            return super().forward(x.view(-1, x.shape[-1])).view(shape) 

