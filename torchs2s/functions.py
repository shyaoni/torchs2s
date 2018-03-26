import numpy as np

import torch
import torch.multiprocessing as mp

from torchs2s.helper import TensorHelper
from torchs2s.tuplize import tuplizer as tur
import torchs2s.utils as utils 

import collections

from IPython import embed

def rnn(cell, inputs=None,
        init_hidden=None,
        helper=None,
        max_lengths=None,
        reduced_fn=None,
        num_workers=1,
        queue=None):
    """Run cell recursively, suitable for both decoding and encoding.
    Args:
        cell: torch.nn.Module. The cell to be run.
        init_hidden (optional): The initial states of the cell. 
                                If not provided, cell.init_hidden() is used.                   
        inputs (optional): Tensor(s). The inputs for the rnn, each takes shape
                           (times, batch_size, ...).
                           It will be used only when helper is not given.
        helper (optional): Use helper to convert the output at current step
                           to the input at next step.
                           It must be declared if inputs is not given.
        max_lengths (optinal): The max step of each sample can be token. 
                               Only used when helper is None.
        reduced_fn (optional): A callable to determine the time to reduce 
                               batch_size. It will be called by
                                 reduced_fn(batch_size, num_finished) 
                               and return `True` if reduce or otherwise `False`.

                               You can use built-in function by the str of 
                               its name, see torchs2s.function.reduced_fn for
                               details.
                               If not given, batch_size is reduced whenever
                               the number of finished samples is more than half.
        num_workers (optional): The number of threads for multiprocessing, 
                                default set to 1.
                                Notices that rnn still runs in one device. If
                                you want to use multi-GPUs, wrap the module
                                class with torch.nn.DataParallel.
    
          
        queue: cannot be used explicitly. 

        NOT IMPLEMENTED YET: num_workers, reduced_fn

        num_workers can only be 1 now! (before pytorch 0.4)
    """
     
    if helper is None:
        helper = TensorHelper(inputs, lengths=max_lengths)

    batch_size = helper.batch_size

    if init_hidden is None:
        init_hidden = cell.init_hidden()
        init_hidden = tur.stack([init_hidden, ] * batch_size, dim=0) 
    
    if num_workers > 1:
        raise NotImplementedError
        if not isinstance(helper, collections.Sequence):
            helpers = helper.split(num_workers)
        else:
            helpers = helper 

        queues, processes = [], []
        for i in range(num_workers):
            q = mp.SimpleQueue()
            p = mp.Process(target=rnn, args=(cell, init_hidden, None,
                                             helpers[i], max_lengths, 
                                             reduced_fn, 1, q))
            processes.append(p)
            queues.append(q)

        for p in processes:
            p.start()

        outputs_g, final_states_g, lengths_g = [], [], []

        for q in queues:
            outputs, final_states, lengths = q.get()
            outputs_g.append(outputs)
            final_states_g.append(final_states)
            lengths_g.append(lengths_g)

        for p in processes:
            p.join()
            
        outputs = tur.cat(outputs_g, dim=1)
        final_states = tur.cat(final_states, dim=0)
        lengths = tur.cat(lengths_g, dim=0)
    
        return outputs, final_states, lengths

    index_order = list(range(batch_size))

    outputs = []
    lengths = torch.LongTensor([0, ] * batch_size)
    final_states = [0, ] * batch_size
    
    cur_outputs, not_finished = [], list(range(batch_size)) 
    output, hidden, step = None, init_hidden, 0

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

        if len(not_finished) <= cur_batch_size // 2: 
            if len(not_finished) == 0:
                hidden = None
            else:
                hidden = tur.get(hidden, not_finished) 
                output = tur.get(output, not_finished)
            
            cur_outputs = tur.expand_to(
                tur.stack(cur_outputs, dim=0), 1, batch_size) 

            r_index = np.argsort(index_order).tolist()

            outputs.append(tur.get(cur_outputs, slice(None), r_index))
            cur_outputs = []

            index_remain, index_subbed = utils.list_sub(
                list(range(cur_batch_size)), not_finished)

            index_order = np.array(index_order)[index_subbed].tolist() + \
                          np.array(index_order)[index_remain].tolist() + \
                          index_order[cur_batch_size:]

            not_finished = list(range(len(not_finished)))

    outputs = tur.cat(outputs, dim=0)
    final_states = tur.stack(final_states, dim=0)

    if queue is None:
        return outputs, final_states, lengths
    else:
        # To do this, we require pytorch 0.4
        queue.put((outputs, final_states, lengths))

if __name__ == '__main__':
    pass
