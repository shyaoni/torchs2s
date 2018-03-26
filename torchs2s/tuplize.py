import torch
import functools
import collections

import torch
from torch.nn import Parameter
from torch.autograd import Variable
from IPython import embed

def is_tensor(tensor):
    return isinstance(tensor, Variable) or \
           isinstance(tensor, torch.FloatTensor) or \
           isinstance(tensor, torch.LongTensor)

class Tuplize(): 
    def __init__(self):
        self.cache = {}
        
    def __getattr__(self, name): 
        if name not in self.cache:
            self.cache[name] = functools.partial(self.func, name)
        return self.cache[name]

    @staticmethod
    def views(tensors, *args):
        if is_tensor(tensors):
            shape_to = []
            for p in args:
                if isinstance(p, slice):
                    shape_to += list(tensors.shape[p])
                elif isinstance(p, collections.Sequence):
                    shape_to += p
                else:
                    shape_to += [p]
            return tensors.view(shape_to)
        elif isinstance(tensors, collections.Sequence):
            return [Tuplize.views(t, *args) for t in tensors]
        else:
            raise ValueError
            

    @staticmethod
    def func(name, tensors, *args, **kwargs): 
        if hasattr(torch.Tensor, name):
            if is_tensor(tensors):
                return getattr(tensors, name)(*args, **kwargs)
            elif isinstance(tensors, collections.Sequence): 
                return [getattr(t, name)(*args, **kwargs) for t in tensors] 
            else:
                raise ValueError
        else:
            if is_tensor(tensors):
                return getattr(torch, name)(tensors, *args, **kwargs)
            elif isinstance(tensors, collections.Sequence):
                if isinstance(tensors[0], collections.Sequence):
                    return [getattr(torch, name)(t, *args, **kwargs) 
                        for t in list(zip(*tensors))]
                else:
                    return getattr(torch, name)(tensors, *args, **kwargs)
            else:
                raise ValueError
         
    @staticmethod
    def flatten(tensors, s, e): 
        if is_tensor(tensors):
            return tensors.view(tensors.shape[:s] + (-1,) + tensors.shape[e:])
        elif isinstance(tensors, collections.Sequence):
            return [t.view(t.shape[:s] + (-1,) + t.shape[e:]) for t in tensors]
        else:
            raise ValueError

    @staticmethod
    def len(tensors, dim):
        if is_tensor(tensors):
            return tensors.shape[dim]
        elif isinstance(tensors, collections.Sequence):
            return tensors[0].shape[dim]
        else:
            raise ValueError

    @staticmethod
    def get(tensors, *args):
        if is_tensor(tensors):
            return tensors[args]
        elif isinstance(tensors, collections.Sequence):
            return [t[args] for t in tensors]
        else:
            raise ValueError 

    @staticmethod
    def expand_to(tensors, dim, num):
        if is_tensor(tensors):
            lst = list(tensors.shape)
            if num == lst[dim]:
                return tensors

            lst[dim] = num - lst[dim]

            if torch.__version__ >= '0.4':
                print('pytorch 0.4 seems to combine tensor and variable. '
                      'Please implement the logic for expand_at manually.')
                raise NotImplementedError
            
            data = tensors
            if not isinstance(data, torch.Tensor):
                data = data.data

            padding_part = type(data)(*lst)
            padding_part.fill_(0)
            if isinstance(tensors, Variable):
                padding_part = Variable(padding_part)

            return torch.cat([tensors, padding_part], dim=dim) 
        elif isinstance(tensors, collections.Sequence):
            return [Tuplize.expand_to(t, dim, num) for t in tensors] 

tuplizer = Tuplize()

if __name__ == '__main__':  
    a = Variable(torch.randn(4, 3).cuda())
    b = Variable(torch.randn(4, 3).cuda())
    d = Variable(torch.zeros(4, 3).cuda())
    e = Variable(torch.zeros(4, 3).cuda())
    f = (a, b)
    g = (d, e)
    r = tuplizer.cat([f, g],dim = 0)

    p = tuplizer.expand_to((a, b), 1, 10)
    print(r)
    print(p)

