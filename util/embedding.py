#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
import numpy as np
from tqdm import tqdm
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class EnEmbedding(nn.Module):
    def __init__(self, spacy_model='en_vectors_web_lg'):
        super().__init__()

        nlp = spacy.load(spacy_model) 
        enbeddings = torch.from_numpy(nlp.vocab.vectors.data)
        n_vocab, vocab_dim  = nlp.vocab.vectors.shape
        self.voc_dim        = vocab_dim
        self.voc_emb        = nn.Embedding(n_vocab, vocab_dim)
        self.voc_emb.weight = nn.Parameter(enbeddings)
        self.voc_emb.weight.requires_grad = False

        return

    def forward(self, voc):
        voc = self.voc_emb(voc[:, 0, :])
        return voc

class ZhEmbedding(nn.Module):
    def __init__(self, f_char_emb, f_word_emb):
        super().__init__()
        self.char_emb, self.char_dim = self.load(f_char_emb)
        self.word_emb, self.word_dim = self.load(f_word_emb)
        self.voc_dim = self.char_dim + self.word_dim
        return

    def load(self, femb):
        emb_np = np.load(femb)
        emb_np = np.append(emb_np, np.zeros((1, emb_np.shape[1])), axis=0)
        emb_nn = nn.Embedding(emb_np.shape[0], emb_np.shape[1])
        emb_nn.weight = nn.Parameter(torch.from_numpy(emb_np).float())
        emb_nn.weight.requires_grad = False
        return emb_nn, emb_np.shape[1]

    def forward(self, voc):
        char = self.char_emb(voc[:, 0, :])
        word = self.word_emb(voc[:, 1, :])
        voc  = torch.cat((char, word), dim=2)  # batch x time x (char_emb + word_emb)
        return voc

class BertEnVocFeatures(nn.Module):
    """
        This class is parallel to EnEmbedding in cnn module
        Because BERT has its own token' embedding, this class does nothing.
    """
    def __init__(self):
        super().__init__()
        return

    def forward(self, voc):
        return voc[:, 0, :]