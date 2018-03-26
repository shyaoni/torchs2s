from torchs2s.rnn import RNNCell, RNN

import torch

from IPython import embed

import numpy as np

torch.cuda.set_device(1)

if __name__ == '__main__':
    x = RNNCell(100, 64)
    model = RNN(x)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
 
    step = 0
    while True:
        inputs = torch.autograd.Variable(torch.randn(100, 64, 100).cuda())
        lengths = torch.from_numpy(np.random.randint(100, size=(64, )))
        lengths[0] = 100
        outputs, final_states, lengths = model(inputs, max_lengths=lengths,
                                               num_workers=1)

        cnt = 0
        mask = torch.zeros(100, 64).cuda() 
        for i, l in enumerate(lengths):
            mask[:l, i] = 1
            cnt += l

        mask = torch.autograd.Variable(mask)

        loss = ((inputs[:, :, :64] - outputs)**2).sum(dim=2) * mask
        loss = loss.sum() / cnt
        
        print(step, loss)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        step += 1
