from torchs2s.rnn import RNNCell, RNN
from torchs2s.utils import mask

import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

torch.cuda.set_device(1)

if __name__ == '__main__':
    model = torch.nn.RNN(100, 64)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
 
    step = 0

    from time import time

    t = 0
    for i in range(1000):
        inputs = torch.autograd.Variable(torch.randn(100, 64, 100).cuda())
        lengths = torch.from_numpy(np.random.randint(100, size=(64, )) + 1)
        lengths[0] = 100

        lengths, indices = lengths.sort(descending=True)

        s = time()
        outputs, states = model(pack_padded_sequence(inputs, lengths.tolist()))


        outputs = pad_packed_sequence(outputs)[0]

        cnt = 0

        msk = torch.autograd.Variable(mask(lengths).cuda())
        loss = ((inputs[:, :, :64] - outputs)**2).sum(dim=2) * msk
        loss = loss.sum() / msk.sum()
        
        print(step, loss)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        t += time() - s

        step += 1

    print(t / 1000)
