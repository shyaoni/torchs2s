import torch
import torch.nn as tnn
import torch.functional as tfunc
from torch.autograd import Variable

from torchs2s.functions import rnn
from torchs2s.tuplize import tuplizer as tur 

from IPython import embed

class RNN(tnn.Module):
    def __init__(self, cell, reduced_fn=None, num_workers=None):
        super().__init__()
        self.cell = cell
        
        self.reduced_fn = reduced_fn
        self.num_workers = num_workers
        
        if self.num_workers is None:
            self.num_workers = 1

    def forward(self, inputs=None, 
                init_hidden=None,
                helper=None,  
                max_lengths=None,
                reduced_fn=None,
                num_workers=None):
        if reduced_fn is None:
            reduced_fn = self.reduced_fn
        if num_workers is None:
            num_workers = self.num_workers

        outputs, final_hidden, lengths = rnn(self.cell, inputs=inputs,
                                             init_hidden=init_hidden,
                                             helper=helper,
                                             max_lengths=max_lengths,
                                             reduced_fn=reduced_fn, 
                                             num_workers=num_workers)

        return outputs, final_hidden, lengths

class RNNCell(torch.nn.RNNCell):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__(input_size, hidden_size, bias, nonlinearity)

        self.register_buffer('default_init_hidden',
                             Variable(torch.zeros(hidden_size)))

    def forward(self, x, hidden):
        h = super().forward(x, hidden)
        return h, h

    def init_hidden(self):
        return self.default_init_hidden

"""
class HierarchicalRNN(HierarchicalRNN):
    def __init__(self, encoder_minor, encoder_major): 
        self.encoder_minor = encoder_minor
        self.encoder_major = encoder_major
        
    def forward(self, init_hidden=None,
                inputs=None,
                helper=None,
                reduced_fn=None,
                num_workers=None,
                **kwargs):
        
         [minor_times, major_times, batch_size, dim, ...]
        encoder_minor_input = 

        self.encoder_minor(
              
            kwargs.get('num_workers_minor', num_workers),
            kwargs.get('num_workers_major', num_workers)
        kwargs.
        if reduced_f
"""
