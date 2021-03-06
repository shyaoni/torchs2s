from torchs2s.rnn import RNNCell, RNN
from torchs2s.utils import mask

import torch

import torchs2s.functions as func

import numpy as np

torch.cuda.set_device(0)

if __name__ == '__main__':
    model = RNN(RNNCell(100, 64))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
 
    step = 0

    from time import time
    
    t = 0
    for i in range(100):
        inputs = torch.autograd.Variable(torch.randn(100, 64, 100).cuda())
        lengths = torch.from_numpy(np.random.randint(100, size=(64, )) + 1)
        lengths[0] = 100

        lengths, indices = lengths.sort(descending=True)

        s = time()
        outputs, final_states, lengths = model(inputs, max_lengths=lengths,
                                               reduced_fn='every')
        t += time() - s

        cnt = 0

        msk = torch.autograd.Variable(mask(lengths).cuda())
        loss = ((inputs[:, :, :64] - outputs)**2).sum(dim=2) * msk
        loss = loss.sum() / msk.sum()
        
        print(step, loss)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        step += 1

    print(func.t / 100)

    print(t / 100)
