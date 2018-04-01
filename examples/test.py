from torchs2s.rnn import RNNCell, RNN
from torchs2s.utils import mask

import torch


import numpy as np

torch.cuda.set_device(1)

if __name__ == '__main__':
    model = RNN(RNNCell(100, 64))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
 
    step = 0
    while True:
        inputs = torch.autograd.Variable(torch.randn(100, 64, 100).cuda())
        lengths = torch.from_numpy(np.random.randint(100, size=(64, )))
        lengths[0] = 100

        lengths, indices = lengths.sort(descending=True)

        outputs, final_states, lengths = model(inputs, max_lengths=lengths,
                                               reduced_fn='every')

        cnt = 0

        msk = torch.autograd.Variable(mask(lengths).cuda())
        loss = ((inputs[:, :, :64] - outputs)**2).sum(dim=2) * msk
        loss = loss.sum() / msk.sum()
        
        print(step, loss)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        step += 1
