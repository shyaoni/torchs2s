import collections
import torchs2s.utils as utils
from torchs2s.data import Field

from IPython import embed

class VocabCounter(collections.Counter):
    def update_one(self, s):
        self.update([s])

class Vocab():
    def __init__(self, datasets, specials=None,
                 max_size=None,
                 min_freq=1):
        if specials is None:
            specials = [] 

        pad_tokens = []
        unk_tokens = []
        for dataset in datasets:
            if isinstance(dataset, Field):
                field = dataset
                specials += field.specials
                pad_tokens.append(field.pad_token)
                unk_tokens.append(field.unk_token)
            
        specials = list(collections.Counter(specials).keys())
        self.pad_token = pad_tokens[0] 
        self.unk_token = unk_tokens[0]

        # build vocab

        c = VocabCounter()
        for dataset in datasets:  
            if isinstance(dataset, Field):
                continue
            elif isinstance(dataset, tuple):
                dataset, tracker = dataset
                utils.capture_and_process(
                    dataset, tracker[0], tracker[1],
                    capture=str,
                    process=c.update_one,
                    num_workers=1)
            else:
                c.update(dataset)

        for spec in specials:
            c.pop(spec, 0)

        tokens = [token for token, freq in c.most_common(max_size)
                            if freq >= min_freq]

        specials, _ = utils.list_sub(
            specials, pad_tokens + unk_tokens)

        self.itos = [self.pad_token] + [self.unk_token] + specials + tokens
        self.stoi = {s: i for i, s in enumerate(self.itos)} 

        for i in unk_tokens:
            self.stoi[i] = 1
        for i in pad_tokens:
            self.stoi[i] = 0

        self.slice_specials = slice(2, 2+len(specials))
        self.slice_tokens = slice(2+len(specials),None)
        self.size = len(self.itos)

    @property
    def specials(self):
        return self.itos[self.slice_specials]

    @property
    def tokens(self):
        return self.itos[self.slice_tokens]

    @property
    def vocab(self):
        return self.itos 

    @property
    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.itos) 

    def __call__(self, handles):
        if isinstance(handles, tuple):
            handles = [handles]

        for handle in handles:
            utils.capture_and_process(handle[0], handle[1][0], handle[1][1],
                                      capture=str,
                                      process=lambda s: self.stoi[s],
                                      clone=True)
                                      

if __name__ == '__main__':
    pass
