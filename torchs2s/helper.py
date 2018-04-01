import numpy as np

import torch
import torch.nn as tnn
import torch.nn.functional as tfunc
from torch.autograd import Variable

import torchs2s.utils as utils
from torchs2s.tuplize import tuplizer as tur

from IPython import embed

class Helper():
    def __init__(self, lengths=None, batch_size=None):
        self.set_lengths(lengths, batch_size)

    def set_lengths(self, lengths=None, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        self.lengths = lengths
        if isinstance(self.lengths, torch.Tensor):
            self.lengths = lengths.tolist()
            self.batch_size = len(lengths)
        elif isinstance(self.lengths, int): 
            self.lengths = [self.lengths,] * self.batch_size
        elif self.lengths is None:
            self.lengths = [inputs.shape[0],] * self.batch_size


class TensorHelper(Helper):
    """A helper that use prior inputs to feed the rnn, usually used by encoders,
    and decoders in training.
    Args:
        inputs: Tensor(s) that each takes shape (times, batch_size, ...).
        lengths (optional): An 1-D list, tensor, np.ndarray with length of
                            batch_size.
                            Or an integer applied to the whole batch.
    """
    def __init__(self, inputs, lengths=None):
        self.inputs = inputs
        self.batch_size = tur.len(inputs, 1)
        super().__init__(inputs.shape[0] if lengths is None else lengths)
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
        if isinstance(output, int):
            index = self.index
        else:
            index = self.index[-tur.len(output, 0):]

        finished = []
        for p, i in enumerate(sorted(index)):
            if self.lengths[i] <= step:
                finished.append(p) 

        return finished, tur.get(self.inputs[step-1], sorted(index))

class SoftmaxHelper(Helper):
    """A helper for decoder during inference. In default, It uses dot product 
    to compute the similarities, and then apply softmax to compute the 
    distribution D. The sum over D is then regarded as the next input.
    Args:
        embedding: The embedding matrix to use, should be [vocab_size, dims].
        ending_idx: The end-of-sentence token's id.
        lengths (optional): An 1-D list, tensor, np.ndarray with length of
                            batch_size.
                            Or an integer applied to the whole batch.

    """
    def __init__(self, embedding, bos_idx=2, eos_idx=3, lengths=None):
        self.embedding_T = embedding.transpose(0, 1).contiguous()
        self.embedding = embedding
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        super().__init__(int(1e8) if lengths is None else lengths, 1)

    def next(self, output, step=1): 
        if isinstance(output, int):
            return [], torch.stack([self.embedding[self.bos_idx],] * output, 0)
  
        batch_size = output.shape[0]

        dist = tfunc.softmax(torch.mm(output, self.embedding_T), dim=1)
        next_input = torch.mm(dist, self.embedding)

        # get finished samples
        pred = dist.max(dim=1)[1]
        if isinstance(pred, Variable):
            pred = pred.data

        finished = []
        batch_size = output.shape[0]

        for i in range(batch_size):
            if len(self.lengths) > 1 and self.lengths[i] <= step:
                finished.append(i)
            elif len(self.lengths) == 0 and self.lengths[0] <= step:
                finished.append(i)
            elif pred[i] == self.eos_idx:
                finished.append(i)

        return finished, next_input
