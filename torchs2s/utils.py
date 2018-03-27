import numpy as np

import torch
from collections import Sequence
from copy import copy
from inspect import isclass
from random import randint

import multiprocessing as mp

from IPython import embed

def is_sequence(x):
    return isinstance(x, list) or isinstance(x, tuple)

def list_sub(lst, removed):
    remain, subbed = [], []
    for x in lst:
        if x not in removed:
            remain.append(x)
        else:
            subbed.append(x)
    return remain, subbed

class TrackerState():
    def __init__(self, nodes, status):
        self.status = status 
        if status == -1 or status == 0 and len(nodes) == 0:
            self.status = -1
            return

        self.nodes = nodes

    def next(self, name):
        new_nodes = []
        new_status = 0
        for node in self.nodes:
            if name in node:
                if node[name] == False:
                    new_status = -1
                    break
                elif node[name] == True:
                    new_status = 1
                else:
                    new_nodes.append(node[name])
            if None in node:
                if node[None] == False:
                    new_status = -1
                    break
                elif node[None] == True:
                    new_status = 1
                else:
                    new_nodes.append(node[name])

        if new_status == 0:
            new_status = self.status
        return TrackerState(new_nodes, new_status)

class Tracker():
    def __init__(self, index, black_index=None):
        if not is_sequence(index) or isinstance(index, tuple):
            if not isinstance(index, tuple):
                index = (index, )
            index = [index]
        if not is_sequence(black_index) or \
           isinstance(black_index, tuple):
            if black_index is not None:
                black_index = [black_index]

        self.tree = {}

        for rule in index:
            self.add_white_rule(rule)

        if black_index is None:
            return

        for rule in black_index:
            self.add_black_rule(rule)

    def add_white_rule(self, rule):
        node = self.tree
        for i in rule[:-1]:
            if i not in node:
                node[i] = {}   
            node = node[i]
        node[rule[-1]] = True

    def add_black_rule(self, rule):
        node = self.tree
        for i in rule[:-1]:
            if i not in node:
                node[i] = {}
            if node[i] == True:
                node[i] = {None: True}
            node = node[i]
        node[rule[-1]] = False 

    def root(self):
        return TrackerState([self.tree], 0)
        
class Capturer():
    def __init__(self, capture, num_workers, index_path, clone):
        self.capture = capture
        self.num_workers = num_workers
        self.index_path = index_path
        self.clone = clone

        self.q = [[] for _ in range(num_workers)]

    def _recursive_capture(self, root, index, tracker):
        if tracker.status == -1:
            return

        if is_sequence(root) or isinstance(root, dict):
            data = root[index]
        else:
            data = root.__dict__[index]

        self.index.append(index)
        if isinstance(data, self.capture):
            if self.index_path:
                self.put(root, index, copy(self.index[1:]))
            else:
                self.put(root, index, None)
        elif is_sequence(data):
            if isinstance(data, tuple):
                root[index] = list(data)
            for i, x in enumerate(data):
                self._recursive_capture(data, i, tracker.next(i))
        elif isinstance(data, dict):
            for k in data.keys():
                self._recursive_capture(data, k, tracker.next(k)) 
        elif isclass(type(data)):
            for k in data.__dict__.keys():
                self._recursive_capture(data, k, tracker.next(k))
        self.index.pop()

    def recursive_capture(self, root, index, tracker):
        self.index = []
        self._recursive_capture(root, index, tracker)

    def put(self, *args):
        i = randint(0, self.num_workers-1)
        self.q[i].append(args)

    def run(self, process):
        if self.num_workers == 1:
            for root, index, data in self.q[0]:
                if data is None:
                    if self.clone:
                        root[index] = process(root[index])
                    else:
                        process(root[index])
                else:
                    process(data)
        else: 
            def pr(q):
                for root, index, data in q: 
                    root[index] = process(root[index])
                    print(root[index])
            ps = []
            for q in self.q:
                ps.append(mp.Process(target=pr, args=(q,)))
            for p in ps:
                p.start()
            for p in ps:
                p.join()
    
def capture_and_process(root, index=None,
                        black_index=None,
                        capture=None,
                        process=None,
                        num_workers=1,
                        clone=False,
                        index_path=False):
    tracker = Tracker(index, black_index)

    capturer = Capturer(capture, 
                        num_workers=num_workers, index_path=index_path,
                        clone=clone)

    capturer.recursive_capture([root], 0, tracker.root()) 

    if process is not None:
        capturer.run(process)

    return root, (index, black_index)

if __name__ == '__main__':
    pass
