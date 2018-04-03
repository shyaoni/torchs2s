from torchs2s.tuplize import tuplizer as tur
from torchs2s.utils import is_sequence
import torchs2s.capture as capture

import torch
from torch.utils.data.dataloader import default_collate

import numpy as np

tensor_type = {
    int:torch.LongTensor,
    float:torch.FloatTensor
}

class PackedSeq():
    def __init__(self, v, level=None):
        self.v = v 
        self._level = 0

        if level is None:
            p = v
            while is_sequence(p):
                p = p[0]
                self._level += 1
        else:
            self._level = level

    def dialog(self):
        self.level(2)
        return self

    def sentence(self):
        self.level(1)
        return self
    
    def level(self, num=None):
        if num is None:
            return self._level

        while self._level < num:
            self._level += 1
            self.v = [self.v]
        while self._level > num:
            self._level -= 1
            self.v = self.v[0]

        return self

    def shape(self):
        if self._level == 1:
            return np.array([len(self.v)])
        elif self._level == 2:
            return np.array([len(self.v), max(len(v) for v in self.v)])

    def lengths(self, shape):
        if self._level == 1:
            return (len(self.v), )
        elif self._level == 2:
            return (tur.expand_to(
                torch.LongTensor([len(v) for v in self.v]), 0, shape[0]), 
                len(self.v))
        
    def tensor(self, shape):
        if self._level == 1:
            if is_sequence(self.v):
                v = tensor_type[type(self.v[0])](self.v)
            else:
                v = self.v
            return tur.expand_to(v, 0, shape[0])  
        elif self._level == 2:
            if is_sequence(self.v):
                if is_sequence(self.v[0]):
                    ttype = tensor_type[type(self.v[0][0])]
                    v = [ttype(x) for x in self.v]
                else:
                    v = self.v
                v = torch.stack(tur.expand_to(v, 0, shape[1]), 0)
            else:
                v = self.v
            return tur.expand_to(v, 0, shape[0])

def get_shape_of_packed_seq(seqs):
    shape = seqs[0].shape()
    for seq in seqs[1:]:
        shape = np.maximum(shape, seq.shape())
    return shape.tolist()

def depadding(tensor, lengths):
    lsts = []
    if lengths.dim() == 1:
        lengths = lengths.tolist()
        batch_size = tensor.shape[1]
        tensor = tensor.transpose(0, 1).contiguous() 
        for i in range(batch_size):
            lsts.append(tensor[i, :lengths[i]].tolist())
    elif lengths.dim() == 2:
        lengths = lengths.transpose(0, 1).tolist()
        batch_size, num_utts = tensor.shape[2], tensor.shape[1]
        tensor = tensor.transpose(0, 2).contiguous() 
        for i in range(batch_size):
            lst = []
            for j in range(num_utts): 
                if lengths[i][j] == 0:
                    break
                lst.append(tensor[i, j, :lengths[i][j]].tolist())
            lsts.append(lst)
    return lsts

def collate(samples):
    batch_size = len(samples)

    handles = capture.get_handle([samples], PackedSeq, (None, '*'))
    capture.process_handle(handles, None, False) # tuple -> list
    handles = [handle[1:] for handle in handles[1]]

    for index in handles: 
        seqs = []
        for sample in samples:
            for i in index:
                sample = sample[i]
            seqs.append(sample)
        shape = get_shape_of_packed_seq(seqs)
        for sample in samples:
            for i in index[:-1]:
                sample = sample[i]
            data = sample[index[-1]]
            sample[index[-1]] = (data.tensor(shape), ) + data.lengths(shape)

    batch = default_collate(samples)

    for index in handles: # transpose the data to time major.
        p = batch
        for i in index:
            p = p[i]

        if len(p) == 2:
            p[0] = p[0].transpose(0, 1).contiguous()
        elif len(p) == 3:
            p[0] = p[0].transpose(0, 2).contiguous()
            p[1] = p[1].transpose(0, 1).contiguous()

    return batch

if __name__ == '__main__':
    pass
