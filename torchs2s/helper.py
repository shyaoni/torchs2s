import numpy as np

import torch
import torch.nn as tnn
import torch.functional as tfunc

import torchs2s.utils as utils
from torchs2s.tuplize import tuplizer as tur

from collections import Sequence

from IPython import embed

class Helper():
    def __init__(self):
        pass

    def next(self, output=None, step=None):
        raise NotImplementedError

    def split(self, num_workers):
        raise NotImplementedError 

class TensorHelper(Helper):
    """A helper that use prior inputs to feed the encoder. 
    Args:
        inputs: Tensor(s) that each takes shape (times, batch_size, ...).
        lengths (optional): An 1-D list, tensor, np.ndarray with length of
                            batch_size.
                            Or an integer applied to the whole batch.
    """
    def __init__(self, inputs, lengths=None):
        self.inputs = inputs
        self.lengths = lengths
        self.pos = 0

        batch_size = tur.len(inputs, 1)

        if isinstance(self.lengths, torch.Tensor):
            self.lengths = lengths.tolist()
        elif isinstance(self.lengths, int): 
            self.lengths = [self.lengths,] * batch_size
        elif self.lengths is None:
            self.lengths = [inputs.shape[0],] * batch_size

        self.batch_size = batch_size
        self.index = np.argsort(self.lengths).tolist()
   
    def next(self, output=None, step=1):
        """
        Args: 
            output (optional): Tensor(s) with shape (batch_size, ...). Use it 
                               to determine current batch_size.
            step (default 0): Int.
            
        Returns: 
            finished: A List[Int] contains the index of finished sample w.r.t. 
                                  current batch_size.
            next_input: Tensor(s). 

        If current batch_size = n, the inputs will be considered selected into 
        shape (times, n, ...) which includes the samples with n-largest lengths
        with order of index. 
        e.g., when batch_size=3 and n=2, where lengths=[5,1,3],
        we select the inputs by order [0,2] along the batch_size dim, or
            selected inputs <- inputs[:, [0,2], ...]

        finished and next_input are then computed w.r.t. the selected inputs.
        """
        if output is None:
            index = self.index
        else:
            index = self.index[-tur.len(output, 0):]

        finished = []
        for p, i in enumerate(sorted(index)):
            if self.lengths[i] <= step:
                finished.append(p) 

        return finished, tur.get(self.inputs[step-1], sorted(index))

    def split(self, num_workers): 
        batch_size = self.batch_size
        step = (self.batch_size - 1) // num_workers + 1
        helpers = []
        for s in range(0, batch_size, step):
            e = min(batch_size, s + step)
            helpers.append(
                TensorHelper(tur.get(self.inputs, slice(None), slice(s, e)), 
                             self.lengths[s:e]))

        return helpers

if __name__ == '__main__':
    from IPython import embed
    
    x = torch.randn((100, 64, 100))
    lengths = np.random.randint(64, size = (64, ))
    helper = TensorHelper(x, lengths=lengths)

    embed()
