import numpy as np

import torch
import torch.multiprocessing as mp

from torchs2s.helper import TensorHelper
from torchs2s.tuplize import tuplizer as tur
import torchs2s.utils as utils 

import collections

from IPython import embed

class ReducedFunc():
    @staticmethod
    def halver(a, b):
        """ halver(batch_size, num_not_finished) <- 
                num_not_finished <= batch_size // 2
        """
        return b <= a//2
    def every(a, b):
        """ every(batch_size, num_not_finished) <- 
                num_not_finished < batch_size
        """
        return b < a

def rnn(cell, inputs=None,
        init_hidden=None,
        helper=None,
        max_lengths=None,
        reduced_fn=None):
    """Run cell recursively, suitable for both decoding and encoding.
    Args:
        cell: torch.nn.Module. The cell to be run.
        init_hidden (optional): The initial states of the cell. 
                                If not provided, cell.init_hidden() is used.                   
        inputs (optional): Tensor(s). The inputs for the rnn, each takes shape
                           (times, batch_size, ...).
                           It will be used to construct a `TensorHelper` 
                           only when helper is not given.
        helper (optional): Use helper to convert the output at current step
                           to the input at next step.
                           It must be declared if inputs is not given.
        max_lengths (optinal): The max step of each sample can be token. 
                               Only used when helper is None.
        reduced_fn (optional): A callable to determine the time to reduce 
                               batch_size. It will be called by
                                 reduced_fn(batch_size, num_not_finished) 
                               and return `True` if reduce or otherwise `False`.

                               You can use built-in function by the str of 
                               its name, see torchs2s.functions.ReducedFunc 
                               for details. If not given, batch_size is 
                               reduced whenever the number of finished samples 
                               is more than half.
    """
    if helper is None:
        helper = TensorHelper(inputs, lengths=max_lengths)

    if init_hidden is None:
        batch_size = helper.batch_size
    else:
        batch_size = tur.len(init_hidden, 0)

    if init_hidden is None:
        init_hidden = cell.init_hidden()
        init_hidden = tur.stack([init_hidden, ] * batch_size, dim=0)

    if reduced_fn is None:
        reduced_fn = ReducedFunc.halver
    elif isinstance(reduced_fn, str):
        reduced_fn = getattr(ReducedFunc, reduced_fn)

    index_order = list(range(batch_size))
    trival_index_order = True

    outputs = []
    lengths = torch.LongTensor([0, ] * batch_size)
    final_states = [0, ] * batch_size
    
    cur_outputs, not_finished = [], list(range(batch_size)) 
    output, hidden, step = batch_size, init_hidden, 0
    while hidden is not None:
        step += 1
        finished, x = helper.next(output, step=step)
        cur_batch_size = tur.len(x, 0)

        not_finished, subbed = utils.list_sub(not_finished, finished)
 
        output, hidden = cell(x, hidden)

        cur_outputs.append(output)

        for i in subbed:
            final_states[index_order[i]] = tur.get(hidden, i)
            lengths[index_order[i]] = step

        if reduced_fn(cur_batch_size, len(not_finished)): 
            if sum(not_finished)*2 == len(not_finished)*(len(not_finished)-1):
                trival_reduced = True
            else:
                trival_reduced = False

            if len(not_finished) == 0:
                hidden = None
            else:
                if trival_reduced:
                    hidden = tur.get(hidden, slice(None, len(not_finished))) 
                    output = tur.get(output, slice(None, len(not_finished)))
                else:
                    hidden = tur.get(hidden, not_finished) 
                    output = tur.get(output, not_finished)
            
            cur_outputs = tur.expand_to(
                tur.stack(cur_outputs, dim=0), 1, batch_size) 

            if trival_index_order:
                outputs.append(cur_outputs)
            else:
                r_index = np.argsort(index_order).tolist()
                outputs.append(tur.get(cur_outputs, slice(None), r_index))
            cur_outputs = []

            index_remain, index_subbed = utils.list_sub(
                list(range(cur_batch_size)), not_finished)

            index_order = np.array(index_order)[index_subbed].tolist() + \
                          np.array(index_order)[index_remain].tolist() + \
                          index_order[cur_batch_size:]
            trival_index_order = trival_index_order & trival_reduced

            not_finished = list(range(len(not_finished)))

    outputs = tur.cat(outputs, dim=0)
    final_states = tur.stack(final_states, dim=0)

    return outputs, final_states, lengths

if __name__ == '__main__':
    pass
