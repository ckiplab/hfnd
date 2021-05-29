#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import spacy

class DummyTokenizer:
    def __init__(self):
        return

    def __call__(self, inst):
        return inst


class EnTokenizer:
    def __init__(self, spacy_model='en_vectors_web_lg', default='*', max_len=100, bagging=False):
        self.nlp     = spacy.load(spacy_model) 
        self.oov     = default
        self.max_len = max_len
        self.bagging = bagging
        return

    def __getitem__(self, voc):
        if not self.nlp.vocab[voc].has_vector: voc = self.oov

        key = self.nlp.vocab.strings[voc]
        row = self.nlp.vocab.vectors.key2row[key]
        return row

    def __call__(self, inst):
        r     = inst['relation']['id']
        head  = inst['head']
        tail  = inst['tail']
        toks  = inst['toks']

        seq_len = len(toks)
        voc     = np.zeros((1, self.max_len), dtype=np.int)
        pos     = np.zeros((7, self.max_len), dtype=np.int)

        for i in range(seq_len):
            voc[0][i] = self[toks[i]]
            for p, e in zip([0, 1], [head, tail]):
                pos_tok = 0
                if i < e['start']: pos_tok = i-e['start']
                elif i > e['end']: pos_tok = i-e['end']
                pos[p][i] = abs(pos_tok) * 2 - (pos_tok < 0) * 1  # map Z to Z+
                #pos[p][i] = pos_tok + max_len

        first, last = head, tail
        if tail['start'] < head['start']: first, last = tail, head

        pos[2][:first['start']]               = 1
        pos[3][first['start']:first['end']+1] = 1
        pos[4][first['end']+1:last['start']]  = 1
        pos[5][last['start']:last['end']+1]   = 1
        pos[6][last['end']+1:seq_len]         = 1

        if self.bagging:
            feature = (r, head['id'], tail['id'], voc, pos)
        else:
            feature = (r, voc, pos)
        return feature