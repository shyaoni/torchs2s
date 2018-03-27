from torchs2s.rnn import GRUCell, HierarchicalRNN, RNN
from torchs2s.data import Field, FieldData, collate
from torchs2s.vocab import Vocab
from torchs2s.utils import mask
from torchs2s.helper import TensorHelper

import torch.nn as tnn
import torch

from torch.autograd import Variable 

from torch.utils.data import DataLoader

from torchs2s.embedder import GloveLoader, Embedder

from IPython import embed

class Dataset():
    def __init__(self, source, target):
        self.dts_src = open(source, 'r').read().splitlines()
        self.dts_tgt = open(target, 'r').read().splitlines()

    def __len__(self):
        return len(self.dts_tgt)

    def __getitem__(self, idx):
        src = FieldData(self.dts_src[idx]).level('dialog')
        tgt = FieldData(self.dts_tgt[idx])
        return src, tgt

class Model(tnn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.encoder = HierarchicalRNN(
            RNN(GRUCell(200, 300)),
            RNN(GRUCell(300, 600)))

        self.decoder = RNN(GRUCell(200, 400))
        self.mediam = tnn.Linear(600, 400)

        loader = GloveLoader(
            200, path='/space/hzt/word_embed/glove/glove.6B.200d.txt') 
        self.embedder = Embedder(vocab, load=loader, 
            init_func=lambda a,b: torch.randn(b), padding_idx=0) 

        self.output_cvt = tnn.Linear(400, 200)

    def forward(self, x, helper=None, **kwargs):
        final_states = self.encoder(self.embedder(x), **kwargs)[1]

        outputs, final_states, lengths = self.decoder(
            init_hidden=self.mediam(final_states),
            helper=helper)

        return self.output_cvt(outputs), final_states, lengths

if __name__ == '__main__':

    field = Field(delimiter='|||',
                  tokenizer=' ')

    dataset = Dataset('../data/source.txt', '../data/target.txt')
    handle = field(dataset)

    vocab = Vocab([field, handle], max_size=10000)
    vocab([handle])
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        collate_fn=collate)

    model = Model(vocab)
    model.embedder = model.embedder

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
            tgt_len = tgt_len - 1

            helper = TensorHelper(target[:-1], tgt_len) 

            outputs = model(
                inputs, helper=helper,
                max_lengths_minor=max_lengths_minor,  
                max_lengths_major=max_lengths_major)[0]

            loss = ((outputs-target[1:])**2).sum(dim=2) * Variable(
                        mask(tgt_len).cuda())
            
            loss = loss.sum() / tgt_len.sum() 


            step += 1
            print('step {} on epoch {}: {}'.format(step, i, loss.data[0])) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
