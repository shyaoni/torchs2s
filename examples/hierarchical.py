import torchs2s.rnn as trnn
from torchs2s.utils import mask

import torch

import numpy as np

torch.cuda.set_device(1)

if __name__ == '__main__':
    minor = trnn.RNN(trnn.RNNCell(100, 100))
    major = trnn.RNN(trnn.LSTMCell(100, 100))

    model = trnn.HierarchicalRNN(minor, major)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    step = 0
    while True:
        inputs = torch.autograd.Variable(torch.randn(100, 10, 30, 100).cuda())
        lengths = torch.from_numpy(np.random.randint(100, size=(10, 30)))
        lengths = 100
        outputs, final_states, lengths = model(inputs, max_lengths=lengths)

        loss = ((inputs[99, :, :, :] - outputs)**2).sum(dim=2)
        loss = loss.sum() / 300 

        print(step, loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
