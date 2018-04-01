import torchs2s.capture as capture

class Field():
    def __init__(self, 
                 bos_token='<bos>',
                 eos_token='<eos>',
                 pad_token='<pad>',
                 unk_token='<unk>',
                 fix_length=None,
                 pad_first=False,
                 trunc_first=False,
                 preprocess=None,
                 postprocess=None,
                 delimiter=None, 
                 tokenizer=None,
                 level=2): # 0 or 1 or 2 
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
        self.level = level
    
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

        if self.level >= 2:
            s = self.delimite(s)
        if self.level >= 1:
            s = [self.tokenize(x) for x in s]
            s = [self.format(x) for x in s] 
        if self.level >= 2:
            if self.delimiter is None:
                s = s[0] 

        if callable(self.postprocess):
            s = self.postprocess(s)

        return s

    def __call__(self, handles, **kwargs):
        if len(kwargs) != 0:
            return self.variant(**kwargs)(dataset, index, clone=clone) 
        
        return capture.process_handle(handles, self.process, True) 

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

