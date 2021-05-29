#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import random
from torch.utils.data import Dataset 
from util.tokenizer import DummyTokenizer

def assure_folder_exist(folder):
    folder = os.path.dirname(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return

class DefaultRetriever:
    def __init__(self):
        return

    def __call__(self, inst):
        info = {
            'id': inst['id'],
            'ep': (inst['head']['id'], inst['tail']['id'])
        }
        return info


class FNSimRetriever:
    def __init__(self):
        return

    def __call__(self, inst):
        info = {
            'id': inst['id'],
            'ep': (inst['head']['id'], inst['tail']['id']),
            'rt': inst['relation']['true_id']
        }
        return info

class RelationDataset(Dataset):
    def __init__(self, fname=None, tokenizer=None, retriever=None):
        super().__init__()
        self.data = [] # [inst, ], in original or tokenized form
        self.info = [] # [info, ], info: {id:, ep:, ...}
        self.bags = {} # {ep: {idx}}

        if fname is not None: 
            self.load(fname, tokenizer=tokenizer, retriever=retriever)
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return idx, self.data[idx]

    def load(self, fname, tokenizer=None, retriever=None):
        with open(fname, 'r') as file: data = json.load(file)
        for inst in data: self.insert(inst, tokenizer=tokenizer, retriever=retriever)
        return self

    def dump(self, fname):
        with open(fname, 'w') as file: json.dump(self.data, file)
        return self

    def insert(self, inst, info=None, tokenizer=None, retriever=None):
        """
            insert: insert a relation learning instance (@inst) to this dataset
            input:
                inst format (basic): {
                    id: identifier (number)
                    relation: {
                        name: relation
                        id  : number
                    }
                    head: {
                        name:  entity name  
                        id  :  number           
                        start: start position
                        end:   end position
                    }
                    tail: {
                        name:
                        id:
                        start:
                        end:
                    }
                    toks: [words]
                }
            output: None
        """ 
        if tokenizer is None: tokenizer = DummyTokenizer()
        if retriever is None: retriever = DefaultRetriever()
        if info is None: info = retriever(inst)
        inst = tokenizer(inst)

        idx = len(self.data)
        ep  = info['ep']
        if ep not in self.bags: self.bags[ep] = set()
        self.bags[ep].add(idx)
        self.data.append(inst)
        self.info.append(info)
        return self

    def split(self, n_fold=0, ratios=[], parts=[]):
        # preprocess n_fold or ratios to parts(subsets of eps)
        if n_fold > 1: ratios = [1 / n_fold] * (n_fold - 1)
        if len(ratios) > 0:
            steps = [0.0]
            for r in ratios: steps.append(steps[-1] + r)
            assert (steps[-1] <= 1.0)

            list_eps = list(self.bags.keys())
            l        = len(list_eps)
            random.shuffle(list_eps)
            parts    = [list_eps[int(l*steps[i]):int(l*steps[i+1])] for i in range(len(ratios))]

        # force every part to be a subset of eps, and add the remained part
        eps = set(self.bags.keys())
        parts = [set(part) & eps for part in parts]  
        remained_eps = eps
        for part in parts: remained_eps -= part
        parts.append(remained_eps)

        datasets = []
        for part in parts:
            dataset = RelationDataset()
            for ep in part:
                for idx in self.bags[ep]: 
                    dataset.insert(self.data[idx], self.info[idx])
            datasets.append(dataset)
            
        return datasets

    def merge(self, datasets=[]):
        for dataset in datasets:
            for ep in dataset.bags:
                for idx in dataset.bags[ep]: 
                    self.insert(dataset.data[idx], dataset.info[idx])

        return self
    """
    def stat_relation(self):
        stat = {}
        if self.tokenized:
            for inst in self.data:
                r = inst[0]
                if r not in stat: stat[r] = 0
                stat[r] += 1

        else: 
            for inst in self.data:
                r = inst['relation']['id']
                if r not in stat: stat[r] = 0
                stat[r] += 1

        for r in stat: stat[r] = stat[r] / len(self.data)
        return stat
    """

def get_stat(dataset):
    n_pos = 0
    n_neg = 0
    for inst in dataset.data:
        r = inst['relation']['id']
        if r == 0: n_neg += 1
        else     : n_pos += 1

    return n_pos, n_neg

def semeval():
    dataset     = 'semeval2010'
    home        = os.path.expanduser("~")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_0.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_0.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

    train       = RelationDataset(ftrain)
    valid       = RelationDataset(fvalid)
    test        = RelationDataset(ftest)

    print(f"{dataset}:")
    print(f"stat(train) = {get_stat(train)}")
    print(f"stat(valid) = {get_stat(valid)}")
    print(f"stat(test)  = {get_stat(test)}")
    print(f"len(total)  = {len(train.data) + len(valid.data) + len(test.data)}")
    return

def tacred():
    dataset     = 'tacred'

    home        = os.path.expanduser("~")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_0.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_0.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

    train       = RelationDataset(ftrain)
    valid       = RelationDataset(fvalid)
    test        = RelationDataset(ftest)

    print(f"{dataset}:")
    print(f"stat(train) = {get_stat(train)}")
    print(f"stat(valid) = {get_stat(valid)}")
    print(f"stat(test)  = {get_stat(test)}")
    print(f"stat(total) = {len(train.data) + len(valid.data) + len(test.data)}")
    return

def sample_inst(dataset, num, fname):
    home        = os.path.expanduser("~")
    fname       = os.path.join(home, f"dataset/{dataset}/{fname}")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train.json")
    train       = RelationDataset(ftrain)

    insts = random.choices(train.data, k=num)
    with open(fname, 'w') as file:
        json.dump(insts, file, ensure_ascii=False, indent=4)

    return


if __name__ == '__main__':
    sample_inst('zhwiki', 100, 'zhwiki_samples.json')
