import collections
import torchs2s.capture as capture
import torchs2s.data as data
from torchs2s.field import Field
from torchs2s.utils import is_sequence, list_sub

class Vocab():
    def __init__(self, handles,
                 max_size=None,
                 min_freq=1):
        specials = [] 

        pad_tokens = [] # place as 0
        unk_tokens = [] # place as 1
        bos_tokens = []
        eos_tokens = []
        for handle in handles:
            if isinstance(handle, Field):
                field = handle
                specials += field.specials
                pad_tokens.append(field.pad_token)
                unk_tokens.append(field.unk_token)
                bos_tokens.append(field.bos_token)
                eos_tokens.append(field.eos_token)
            
        specials = list(collections.Counter(specials).keys())
        self.pad_token = pad_tokens[0] 
        self.unk_token = unk_tokens[0]
        self.eos_token = eos_tokens[0]
        self.bos_token = bos_tokens[0]

        # build vocab

        c = collections.Counter()
        def recursive_add(v):
            if is_sequence(v[0]):
                for x in v:
                    recursive_add(x)
            else:
                c.update(v)
        
        for handle in handles:  
            if isinstance(handle, Field):
                continue
            elif isinstance(handle, tuple): # handle
                capture.process_handle(handle, recursive_add, False)
            else:
                c.update(handle)

        for spec in specials:
            c.pop(spec, 0)

        tokens = [token for token, freq in c.most_common(max_size)
                            if freq >= min_freq]

        specials, _ = list_sub(
            specials, pad_tokens + unk_tokens + eos_tokens[:1] + bos_tokens[:1])

        self.itos = [self.pad_token, self.unk_token, 
                     self.bos_token, self.eos_token] + specials + tokens
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

    def recursive_numericalize(self, v):
        for i, x in enumerate(v):
            if is_sequence(x):
                self.recursive_numericalize(x)
            elif isinstance(x, str):
                v[i] = self.stoi.get(x, 1)     

    def __call__(self, handles): 
        capture.process_handle(handles, self.recursive_numericalize, False)

