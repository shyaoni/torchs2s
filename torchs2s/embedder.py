from tqdm import tqdm
import torch

import torchs2s.utils as utils

from torch.nn import Embedding

class Loader():
    pass

class GloveLoader(Loader):
    def __init__(self, dim, path=None, cache=None):
        self.dim = dim
        self.path = path
        
    def __call__(self, vocab, init_func):
        print('Get embedding weights for {} from {}...'.format(
            vocab, self.path))

        tensor = [0,] * vocab.size

        dim = self.dim
        
        founds = []
        with open(self.path, 'r') as f:
            for line in tqdm(f): 
                vec = line.strip().split()
                if len(vec) == 0:
                    continue
                word, vec = vec[0], vec[1:]
                if word not in vocab.stoi:
                    continue
                if len(vec) != dim:
                    raise ValueError

                idx = vocab.stoi[word]
                tensor[idx] = torch.FloatTensor([float(v) for v in vec])
                founds.append(idx)

        not_found, _ = utils.list_sub(list(range(vocab.size)), founds)
        for i in not_found:
            if init_func is not None:
                tensor[i] = init_func(vocab.itos[i], dim)

        return torch.stack(tensor, dim=0) 

def load_from(vocab, load, init_func):
    if isinstance(load, Loader):
        return load(vocab, init_func)
    else:
        raise NotImplementedError

class Embedder(Embedding):
    def __init__(self, vocab, load='', init_func=None, 
                 **kwargs):
        tensor = load_from(vocab, load, init_func)
        super().__init__(tensor.shape[0], tensor.shape[1], **kwargs)
        self.weight.data.copy_(tensor)

    def forward(self, x):
        if x.dim() == 2:
            return super().forward(x)
        else:
            shape = x.shape + (self.embedding_dim, )
            return super().forward(x.view(-1, x.shape[-1])).view(shape)
             

