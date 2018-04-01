from torchs2s.rnn import GRUCell, HierarchicalRNN, RNN
from torchs2s.field import Field
from torchs2s.data import PackedSeq, collate
from torchs2s.vocab import Vocab
from torchs2s.utils import mask
from torchs2s.helper import TensorHelper
from torchs2s.capture import get_handle
from torchs2s.embedder import Embedder

import torch.nn as tnn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

class Dataset():
    def __init__(self, source, target):
        self.dts_src = open(source, 'r').read().splitlines()
        self.dts_tgt = open(target, 'r').read().splitlines()

    def __len__(self):
        return len(self.dts_tgt)

    def __getitem__(self, idx):
        src = PackedSeq(self.dts_src[idx]).dialog()
        tgt = PackedSeq(self.dts_tgt[idx]).sentence()
        return src, tgt

class Model(tnn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = HierarchicalRNN(
            RNN(GRUCell(200, 300)), RNN(GRUCell(300, 600)))

        self.medium = tnn.Linear(600, 400)
        self.decoder = RNN(GRUCell(200, 400))

        self.output_cvt = tnn.Linear(400, 200)

    def forward(self, x, helper=None, **kwargs):
        encoder_states = self.encoder(self.embedder(x), **kwargs)[1]
        
        outputs, final_states, lengths = self.decoder(
            init_hidden=self.medium(encoder_states),
            helper=helper) 

        return self.output_cvt(outputs), final_states, lengths

if __name__ == '__main__':
    dataset = Dataset('../data/source.txt', '../data/target.txt')
    handle = get_handle(dataset, str)

    field = Field(delimiter='|||', tokenizer=' ')
    field(handle)
    vocab = Vocab([field, handle], max_size=1000)
    vocab(handle)
 
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        collate_fn=collate)

    model = Model()
    model.embedder = Embedder(200, vocab) 
    model.cuda()

    num_epochs = 1000
    step = 0

    optimizer = torch.optim.Adam(model.parameters())
    for i in range(num_epochs):
        for dialog, target in loader:
            inputs, max_lengths_minor, max_lengths_major = dialog
            inputs = Variable(inputs.cuda())

            target, tgt_len = target
            target = Variable(target.cuda())
            target = model.embedder(target)

            helper = TensorHelper(target[:-1], tgt_len - 1)

            outputs = model(inputs, helper=helper,
                max_lengths=max_lengths_minor)[0]

            msk = Variable(mask(tgt_len-1).cuda())

            loss = ((outputs-target[1:])**2).sum(dim=2) * msk
            loss = loss.sum() / msk.sum()

            step += 1
            print('step {} on epoch {}: {}'.format(step, i, loss.data[0])) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
