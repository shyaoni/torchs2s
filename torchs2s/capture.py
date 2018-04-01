#
"""This module implements the core function to capture handles and process by 
address.
"""
import numpy as np

import torch
from collections import Sequence
from torchs2s.utils import is_sequence

global special_char
special_char = '*'

def set_special(x):
    global special_char
    special_char = x

class TrackerState():
    def __init__(self, white, black):
        global special_char 
        self.white = white
        self.black = black
        self.spc = special_char

    def failed(self):
        return self.black == True or self.white == False

    def next(self, index):
        if self.white == True:
            next_white, next_index = True, index
        elif None in self.white:
            next_white, next_index = self.white[None], None
        elif self.spc in self.white:
            next_white, next_index = self.white[self.spc], index
        elif index in self.white:
            next_white, next_index = self.white[index], index
        else:
            next_white, next_index = False, index

        if self.black == False:
            next_black = False
        elif None in self.black:
            next_black = self.black[None]
        elif index in self.black:
            next_black = self.black[index]
        else:
            next_black = False

        return TrackerState(next_white, next_black), next_index

    def none(self):
        return True

class Tracker():
    def __init__(self, white_list=None, black_list=None):
        if not is_sequence(white_list) or isinstance(white_list, tuple):
            if not isinstance(white_list, tuple):
                white_list = (white_list, )
            white_list = [white_list]

        if not is_sequence(black_list) or isinstance(black_list, tuple):
            if black_list is not None:
                if not isinstance(black_list, tuple):
                    black_list = (black_list, )
                black_list = [black_list]

        self.white = {}
        self.black = {}

        for rule in white_list:
            self.add_white_rule(rule)

        if black_list is None: 
            return

        for rule in black_index:
            self.add_black_rule(rule)

    def add_white_rule(self, rule):
        node = self.white
        for i in rule[:-1]:
            if i not in node:
                node[i] = {}   
            node = node[i]

        global special_char
        if rule[-1] == special_char:
            node[rule[-1]] = 2
        else:
            node[rule[-1]] = True

    def add_black_rule(self, rule):
        node = self.black
        for i in rule[:-1]:
            if i not in node:
                node[i] = {}
            node = node[i]
        node[rule[-1]] = True

    def root(self):
        return TrackerState(self.white, self.black)
        
class Capturer():
    """Capturer can be used to derive the very specific handle.
    """
    def __init__(self, capture, tracker):
        self.capture = capture
        self.tracker = tracker

    def recursive_capture(self, data, index, trace):
        if index != -1:
            trace, index = trace.next(index) 
            if trace.failed():
                return
            self.index.append(index)

        if isinstance(data, self.capture):
            p = tuple(self.index)
            if p not in self.captured_index:
                self.captured_index.add(p)
        elif isinstance(data, list):
            if trace.none() == False:
                raise ValueError('capturer failed: list only accepts None.')
            else:
                self.recursive_capture(data[0], None, trace)
        elif isinstance(data, tuple):
            for i, x in enumerate(data):
                self.recursive_capture(x, i, trace)
        elif isinstance(data, dict): 
            for k, v in data.items():
                self.recursive_capture(v, k, trace)
        elif hasattr(data, '__dict__'):
            for k, v in data.__dict__.items():
                self.recursive_capture(v, k, trace)

        if index != -1:
            self.index.pop()

    def __call__(self, root):
        self.captured_index = set()
        self.index = []
        self.recursive_capture(root, -1, self.tracker.root())

        return list(self.captured_index)

def get_handle(roots, capture, white_list=None, black_list=None):
    """Function to get handle.
    Args:
        roots (list, class): The capture will start at each entry of `roots`.
                             If `roots` is a class, it will be automatically
                             converted to [`roots`].
        capture: The type to be captured.
        white_list (optional): A list contains all accepted index paths, where
                               a index path is a tuple specifying the 
                               level-by-level index, e.g., ('name', None, 0) 
                               means that handles are produced by any element
                               with type `capture` in root['name'][*][0]([*])*,
                               where the last bracket follows regular exp.
                               A str index can also be used to get class attr.
                               
        black_list (optional): A list contains all rejected index paths,
                               overriding `white_list`.
                             
    We only accept recursive dict, list, tuple and class with rules:

        dict: any key or None.
        list: must be None and will always go into the first entry only.
        tuple: any index or None.
        class: any attrname or None.
    
    Any real index path cannot be accepted by more than one in white list.

    Returns:
        handle(s): A handle is a tuple (root, white_list, black_list), where
        white_list are those EXACT captured paths, e.g., if dataset.data is a 
        list of str, then get_handle(dataset, str, None) produces
            (root, [(data, None), ], None) as a handle.

        If `roots` contains only one element, one handle is simply returned.
    """
    capturer = Capturer(capture, Tracker(white_list, black_list))

    assert is_sequence(roots) or hasattr(roots, '__dict__')
    if hasattr(roots, '__dict__'):
        roots = [roots]

    handles = [(root, capturer(root), black_list) for root in roots]

    if len(handles) == 1:  
        return handles[0]
    else:
        return handles

class Processer():
    def __init__(self, tracker, process, clone):
        self.tracker = tracker
        self.process = process
        self.clone = clone

    def recursive_process(self, data, index, parent, trace):
        if index != -1:
            trace, _ = trace.next(index) 
        if trace.failed():
            return
        if trace.white == True:
            if self.process is not None:
                if self.clone:
                    parent[index] = self.process(data)
                else:
                    self.process(data)
        elif isinstance(data, list):
            for i, x in enumerate(data):
                self.recursive_process(data[i], i, data, trace)
        elif isinstance(data, tuple):
            parent[index] = list(data) 
            for i, x in enumerate(data):
                self.recursive_process(data[i], x, data, trace)
        elif isinstance(data, dict): 
            for k, v in data.items():
                self.recursive_process(v, k, data, trace)
        elif hasattr(data, '__dict__'):
            for k, v in data.__dict__.items():
                self.recursive_process(v, k, data, trace)

    def __call__(self, root):
        self.recursive_process(root, -1, None, self.tracker.root())

def process_handle(handles, process, clone):
    """Process variables specifying by the handles.
    Args:
        handles: handle(s) produced by `get_handle`.
        process: the process to be called.
        clone (True or False): If True, override elements by x = process(x)
    """
    if not isinstance(handles, list):
        handles = [handles]

    for handle in handles:
        root, white_list, black_list = handle 
        tracker = Tracker(white_list, black_list)
        processer = Processer(tracker, process, clone)
        processer(root)

if __name__ == '__main__':
    pass
