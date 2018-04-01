from torchs2s.field import Field
from torchs2s.data import collate, PackedSeq
from torchs2s.vocab import Vocab
from torchs2s.capture import get_handle

from torch.utils.data import DataLoader

from IPython import embed

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

if __name__ == '__main__':
    field = Field(delimiter='|||', tokenizer=' ')
    
    dataset = Dataset('../data/source.txt', '../data/target.txt') 
    
    handle = get_handle(dataset, str)
    field(handle)
    vocab = Vocab([field, handle], max_size=10000)  
    vocab(handle) 

    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        collate_fn=collate, num_workers=0)  

    cnt = 0
    for i in range(1000):
        print(i)
        for idx, (dialog, target) in enumerate(loader):
            cnt += 1
