from torchs2s.tuplize import tuplizer as tur
import torchs2s.utils as utils


import torch
import collections

from torch.utils.data.dataloader import default_collate

from IPython import embed

levels = ['word', 'utterance', 'dialog']
levels_dict = {s: i for i, s in enumerate(levels)} 

tensor_type = {
    False: {
        int: torch.LongTensor
    },
    True: {
        int: torch.cuda.LongTensor
    }
}

def _level_num(x):
    return levels_dict.get(x, None)

def _level_check(s, level):
    a = _level_num(s)
    b = _level_num(level)
    if a is None or b is not None and a >= b:
        return True
    else:
        return False

class Field():
    # 1. process raw data
    def __init__(self, 
                 bos_token='<bos>',
                 eos_token='<eos>',
                 pad_token='<pad>',
                 unk_token='<unk>',
                 fix_length=None,
                 pad_first=False,
                 trunc_first=False,
                 preprocess=None,
                 postprocess=None, # I think usually its a remapping or a counter
                 delimiter=None, 
                 tokenizer=None,
                 level=None,
                 num_workers=1): 
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.pad_first = pad_first
        self.trunc_first = trunc_first
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.delimiter = delimiter
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.level=level
    
    def variant(self, **kwargs):
        d = copy(self.__dict__)
        d.update(kwargs) 
        return type(self)(**d)

    def delimite(self, s):
        if self.delimiter is None:
            return [s]
        elif isinstance(self.delimiter, str):
            return s.split(self.delimiter)
        else:
            return self.delimiter(s)
    
    def tokenize(self, s): 
        if self.tokenizer is None:
            return [s]
        elif isinstance(self.tokenizer, str):
            return s.split(self.tokenizer)
        else:
            return self.tokenizer(s)

    def format(self, s): 
        if self.fix_length is not None:
            length_to = self.fix_length + (
                self.bos_token, self.eos_token).count(None) - 2

            if len(s) > length_to:
                if self.trunc_first:
                    s = s[-length_to:]
                else:
                    s = s[:length_to]
        
        if self.bos_token is not None:
            s = [self.bos_token] + s 
        if self.eos_token is not None:
            s = s + [self.eos_token]

        if self.fix_length:
            if self.pad_first:
                s = [self.pad_token, ] * (self.fix_length - len(s)) + s
            else:
                s = s + [self.pad_token, ] * (self.fix_length - len(s))

        return s

    def process(self, s):
        if callable(self.preprocess):
            s = self.preprocess(s)

        if _level_check(self.level, 'dialog'):
            s = self.delimite(s)
        if _level_check(self.level, 'utterance'):
            s = [self.tokenize(x) for x in s]
            s = [self.format(x) for x in s] 

        if _level_check(self.level, 'dialog'):
            if self.delimiter is None:
                s = s[0] 

        if callable(self.postprocess):
            s = self.postprocess(s)

        return s 

    def __call__(self, dataset, index=None, black_index=None, clone=True, 
                 **kwargs):
        if len(kwargs) != 0:
            return self.variant(**kwargs)(dataset, index, clone=clone) 

        return utils.capture_and_process(dataset, index, 
                                         black_index=black_index,
                                         capture=str, 
                                         num_workers=self.num_workers, 
                                         process=self.process,
                                         clone=clone)

    @property
    def specials(self): 
        specs = []
        if self.pad_token is not None:
            specs.append(self.pad_token)
        if self.bos_token is not None:
            specs.append(self.bos_token)
        if self.eos_token is not None:
            specs.append(self.eos_token)
        if self.unk_token is not None:
            specs.append(self.unk_token)
        return specs 

class FieldData():
    # pack str data with specific recursive level
    def __init__(self, v, cuda=False):  
        p = v
        r = 0
        while utils.is_sequence(p):
            p = p[0]  
            r += 1
 
        self._level = None
        self._rtime = r
        self.type = type(p)

        self.v = v
        self._cuda = False

        self.level(None)

    def level(self, level=-1):
        if level == -1:
            return self._level
        else:
            self._level = levels[level] if isinstance(level, int) else level

            r = _level_num(self._level)
            v = self.v
            
            if r is None:
                while len(v) == 1 and utils.is_sequence(v[0]):
                   v = v[0] 
                   self._rtime -= 1 
            else:
                while self._rtime < r:
                    v = [v]
                    self._rtime += 1
                while self._rtime > r:
                    v = v[0]
                    self._rtime -= 1
            self.v = v

            return self

    def __str__(self):
        return self.v.__str__()    

    def cuda(self):
        self._cuda = True
        return self
    
    def cpu(self):
        self._cuda = False
        return self

    def tensor(self, shape):
        level = len(shape)
        v = self.v

        if level == 0: 
            return tensor_type[self._cuda][self.type]([v])
        elif level == 1:
            return tur.expand_to(
                tensor_type[self._cuda][self.type](v), 0, shape[0]), len(v)
        elif level == 2:
            return (tur.expand_to(
                torch.stack(
                    tur.expand_to(
                        [tensor_type[self._cuda][self.type](x) for x in v], 
                        0, shape[1]),
                    dim=0),
                0, shape[0]),
                tur.expand_to(
                    torch.LongTensor([len(x) for x in v]), 
                    0, shape[0]),
                len(v))  
        else:
            raise NotImplementedError
 
class FieldDataBatcher():     
    def __init__(self):
        self._level = None
        self.ids = []
    
    def add(self, parent, idx, data):  
        if data.level() is not None:
            if self._level is None or _level_check(data.level(), self._level):
                self._level = data.level()
        self.ids.append((parent, idx, data)) 

    def batch(self):
        level = _level_num(self._level)
        if level is None:
            level = 0

        for parent, idx, data in self.ids: 
            data.level(self._level)
            level = max(level, data._rtime)

        if level == 0:
            shape = []
        elif level == 1:
            shape = [0]
            for parent, idx, data in self.ids:
                shape[0] = max(shape[0], len(data.v))
        elif level == 2:
            shape = [0, 0]
            for parent, idx, data in self.ids:
                shape[0] = max(shape[0], len(data.v))
                shape[1] = max(shape[1], max(len(v) for v in data.v)) 

        for parent, idx, data in self.ids:
            parent[idx] = data.tensor(shape) 

        self.ids = []

def collate(samples):
    batch_size = samples

    q = []
    
    _ = utils.capture_and_process(samples, capture=FieldData)
    _ = utils.capture_and_process(samples[0],
                                  capture=FieldData,
                                  process=q.append,
                                  index_path=True)
                                  
    for index in q:
        batcher = FieldDataBatcher()
        for sid, x in enumerate(samples): 
            if len(index) == 0:
                batcher.add(samples, sid, x)
                continue

            for i in index[:-1]:
                x = x[i]
            batcher.add(x, index[-1], x[index[-1]])
        
        batcher.batch()

    r = default_collate(samples)
    
    for index in q:
        p = r
        for i in index:
            p = p[i]
            
        if utils.is_sequence(p):
            if len(p) == 2:
                p[0] = p[0].transpose(0, 1).contiguous()
            else:
                p[0] = p[0].transpose(0, 2).contiguous()
                p[1] = p[1].transpose(0, 1).contiguous()

    return r

if __name__ == '__main__':
    pass
