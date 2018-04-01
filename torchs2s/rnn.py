import torch
import torch.nn as tnn
import torch.functional as tfunc
from torch.autograd import Variable

from torchs2s.functions import rnn
from torchs2s.tuplize import tuplizer as tur 
from torchs2s.utils import is_tensor

import collections

class RNNBase(tnn.Module):
    def __init__(self, reduced_fn=None):
        super().__init__()
        self.reduced_fn = reduced_fn

    def get_reduced_fn(self, **kwargs):
        return kwargs.get('reduced_fn', self.reduced_fn)

class RNN(RNNBase):
    def __init__(self, cell, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell 

    def forward(self, inputs=None, init_hidden=None, helper=None,  
                max_lengths=None, **kwargs):
        reduced_fn = self.get_reduced_fn(**kwargs)

        outputs, final_hidden, lengths = rnn(self.cell, inputs=inputs,
                                             init_hidden=init_hidden,
                                             helper=helper,
                                             max_lengths=max_lengths,
                                             reduced_fn=reduced_fn)

        return outputs, final_hidden, lengths

class HierarchicalRNN(RNNBase):
    def __init__(self, encoder_minor, encoder_major, medium=None, **kwargs): 
        super().__init__(**kwargs)
        self.encoder_minor = encoder_minor
        self.encoder_major = encoder_major
        self.medium = medium
        
    def forward(self, inputs=None,
                init_hidden=None,
                helper=None,
                max_lengths=None,
                **kwargs):
        if inputs is None:
            raise NotImplementedError

        reduced_fn = self.get_reduced_fn(**kwargs)

        max_lengths = kwargs.get('max_lengths_minor', max_lengths)
        if is_tensor(max_lengths):
            max_lengths = max_lengths.view(-1)

        batch_size = inputs.shape[2] 
        inputs_minor = tur.views(inputs, slice(None, 1), -1, slice(3, None))

        outputs_minor, states_minor, lengths = self.encoder_minor(
            inputs_minor, init_hidden,
            helper=helper,
            max_lengths=max_lengths,
            reduced_fn=kwargs.get('reduced_fn_minor', reduced_fn))


        if self.medium is not None:
            states_minor = self.medium(states_minor) 
        elif isinstance(states_minor, collections.Sequence):
            states_minor = tur.views(states_minor, slice(None,1), -1)  
            states_minor = torch.cat(states_minor, dim=1)

        states_minor = tur.views(
            states_minor, -1, batch_size, slice(1, None)) 

        outputs_major, states_major, lengths = self.encoder_major(
            states_minor, init_hidden,
            max_lengths=kwargs.get('max_lengths_major', None),
            reduced_fn=kwargs.get('reduced_fn_major', reduced_fn))

        return outputs_major, states_major, lengths 

class RNNCell(torch.nn.RNNCell):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__(input_size, hidden_size, bias, nonlinearity)

        self.register_buffer('default_h',
                             Variable(torch.zeros(hidden_size)))

    def forward(self, x, hidden):
        h = super().forward(x, hidden)
        return h, h

    def init_hidden(self):
        return self.default_h

class LSTMCell(torch.nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.register_buffer('default_h',
                             Variable(torch.zeros(hidden_size)))
    
    def forward(self, x, hidden):
        h, c = super().forward(x, hidden)
        return c, (h, c)

    def init_hidden(self):
        return (self.default_h, self.default_h)

class GRUCell(torch.nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias=True)

        self.register_buffer('default_h', 
                             Variable(torch.zeros(hidden_size)))

    def forward(self, x, hidden):
        h = super().forward(x, hidden)
        return h, h

    def init_hidden(self):
        return self.default_h
