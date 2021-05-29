#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
import numpy as np
from tqdm import tqdm
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.data import RelationDataset, assure_folder_exist
from util.embedding import EnEmbedding, ZhEmbedding, BertEnVocFeatures
from util.tokenizer import EnTokenizer, ZhTokenizer, BertEnTokenizer
from util.measure import MicroF1
from base.cnn import ModelCNN, Trainer
from base.rbert import RBert
from denoise.coteaching import CoteachingTrainer

class CleanLab:
    def __init__(self, model, voc_emb, cuda_devices='0'):
        # models
        self.voc_emb  = voc_emb
        self.model    = model
        self.n_class  = model.n_class
        self.cuda     = cuda_devices
        return

    def clean(self, folder, train, valid, n_fold=4, epochs=100, batch=256, lr=0.001):
        prob_train = self.estimate_prob(folder, train, valid, n_fold, epochs, batch, lr)
        # print(prob_train)

        q_table = self.estimate_qtable(train, prob_train)
        # print(q_table)

        train, wgt = self.clean_dataset(train, prob_train, q_table)
        

        #q_table = self.estimate_qtable(valid, prob_valid)
        #valid, _   = self.clean_dataset(valid, prob_valid, q_table)
        
        return train, wgt

    def estimate_prob(self, folder, train, valid, n_fold=4, epochs=100, batch=256, lr=0.001):
        folds = train.split(n_fold=n_fold)
        prob_tmp   = {}
        #prob_valid = 0

        for k in range(n_fold):
            merged = RelationDataset().merge([folds[j] for j in range(n_fold) if j != k])
            left   = folds[k]
            
            self.model.reset_net()
            trainer = Trainer(self.model, self.voc_emb, cuda_devices=self.cuda)
            trainer.train(folder, merged, valid, epochs=epochs, batch=batch, lr=lr)
            probs = trainer.predict(folder, left, batch=batch)
            for i in range(len(left)):
                inst_id = left.info[i]['id']
                prob_tmp[inst_id] = probs[i]

            #prob_valid += trainer.predict(folder, valid, batch=batch)

        #prob_valid /= n_fold
        prob_train = torch.zeros((len(train), self.n_class))
        for i in range(len(train)):
            inst_id = train.info[i]['id']
            prob_train[i] = prob_tmp[inst_id]

        prob_train = prob_train.cpu().numpy()
        #prob_valid = prob_valid.cpu().numpy()

        return prob_train

    def estimate_qtable(self, dataset, probs):
        c_table = np.zeros((self.n_class, self.n_class)) # |rd| x |rt|
        thres   = np.zeros(self.n_class)
        margin  = np.zeros(self.n_class)

        for prob, (rd, _, _, voc, pos) in zip(probs, dataset.data):
            thres[rd] += prob[rd]
            margin[rd] += 1

        thres = thres / margin
        # fixing the nan problem
        thres = np.nan_to_num(thres)

        for prob, (rd, _, _, voc, pos) in zip(probs, dataset.data):
            if rd != 0:
                # force keeping positive
                rt = rd
                c_table[rd][rt] += 1
                continue

            mask = (prob > thres)
            if not np.any(mask): 
                continue
            rt   = np.argmax(prob * mask)
            c_table[rd][rt] += 1

        norm    = margin / np.sum(c_table, axis=1)
        norm = np.nan_to_num(norm)
        n_table = c_table * (norm[:, None]) # multiply by row
        q_table = n_table / np.sum(n_table)
        q_table = np.nan_to_num(q_table)

        return q_table

    def clean_dataset(self, dataset, probs, q_table):
        cleaned = RelationDataset()
        discard = set()
        wgt     = np.zeros(self.n_class) # default zero
        for i in range(self.n_class):
            if q_table[i][i] != 0:
                wgt[i] = np.sum(q_table[:, i]) / q_table[i][i]
        #wgt = np.sum(q_table, axis=0) / np.diag(q_table)

        for rd in range(self.n_class):
            for rt in range(self.n_class):
                if rd == rt: continue
                diffs = []
                for idx, prob in enumerate(probs):
                    if dataset.data[idx][0] == rd:
                        diffs.append((prob[rt] - prob[rd], idx))
                diffs = sorted(diffs, reverse=True)
                n_pruned = min(len(diffs), int(len(probs) * q_table[rd][rt]))
                for d, idx in diffs[:n_pruned]:
                    discard.add(idx)

        for idx in range(len(dataset)):
            #if idx in discard: assert(dataset.data[idx][0] == 0)
            if idx not in discard:   
                cleaned.insert(dataset.data[idx], dataset.info[idx])

        return cleaned, wgt


def count_neg(dataset):
    num = 0
    for r, _, _, voc, pos in dataset.data:
        if r == 0: num += 1

    return num

def train_semeval():
    cuda        = '0'
    dataset     = 'semeval2010'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fns         = range(0, 6)
    n_fold      = 4
    n_relation  = 10
    max_len     = 100
    pre_epochs  = 10
    epochs      = 150
    batch       = 256
    lr          = 0.003
    repeat      = 5
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_cleanlab.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}_{arch}_cleanlab/fn_{fn}/")
                print(folder)
                ftrain = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
                fvalid = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
                ftest  = os.path.join(home, f"dataset/{dataset}/test.json")
                train  = RelationDataset(ftrain, tokenizer=tokenizer)
                valid  = RelationDataset(fvalid, tokenizer=tokenizer)
                test   = RelationDataset(ftest,  tokenizer=tokenizer)
                models = [
                    ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation),
                    ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
                ]
                num_neg  = count_neg(train)
                ratio    = fn * 0.1
                cleanlab = CleanLab(models[0], voc_emb, cuda_devices=cuda)
                train, wgt = cleanlab.clean(folder, train, valid, n_fold=n_fold, epochs=epochs, batch=batch, lr=lr)
                new_neg  = count_neg(train)
                ratio    = max(0, ratio - (new_neg / num_neg))

                models[0].reset_net()
                trainer  = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
                trainer.train(folder, train, valid, ratio=ratio, wgt=wgt, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
                f1, p, r = trainer.test(folder, 0, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return

def train_tacred():
    cuda        = '3'
    dataset     = 'tacred'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fns         = [2, 8]
    n_fold      = 4
    n_relation  = 42
    max_len     = 100
    pre_epochs  = 10
    epochs      = 200
    batch       = 256
    lr          = 0.0003
    repeat      = 1
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_cont_cleanlab_nr.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}_{arch}_cont_cleanlab_nr/fn_{fn}/")
                print(folder)
                ftrain = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
                fvalid = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
                ftest  = os.path.join(home, f"dataset/{dataset}/test.json")
                train  = RelationDataset(ftrain, tokenizer=tokenizer)
                valid  = RelationDataset(fvalid, tokenizer=tokenizer)
                test   = RelationDataset(ftest,  tokenizer=tokenizer)
                models = [
                    ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation),
                    ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
                ]
                num_neg  = count_neg(train)
                ratio    = 0.0 #fn * 0.1
                cleanlab = CleanLab(models[0], voc_emb, cuda_devices=cuda)
                train, wgt = cleanlab.clean(folder, train, valid, n_fold=n_fold, epochs=epochs, batch=batch, lr=lr)
                new_neg  = count_neg(train)
                ratio    = max(0, ratio - (new_neg / num_neg))

                models[0].reset_net()
                trainer  = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
                trainer.train(folder, train, valid, ratio=ratio, wgt=wgt, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
                f1, p, r = trainer.test(folder, 0, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return

def train_nyt():
    cuda        = '0'
    dataset     = 'nyt'
    archs       = ['cnn']
    spacy_model = 'en_vectors_web_lg'
    n_fold      = 4
    n_relation  = 53
    max_len     = 256
    pre_epochs  = 10
    epochs      = 10
    batch       = 256
    lr          = 0.0003
    repeat      = 1
    # fn          = 0.14 ## from sampling: 28/200
    fn          = 1.4  # 0.14 * 10

    home        = os.path.expanduser("~")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

    tokenizer = EnTokenizer(spacy_model=spacy_model, max_len=max_len, bagging=True)
    train = RelationDataset(ftrain, tokenizer=tokenizer)
    valid = RelationDataset(fvalid, tokenizer=tokenizer)
    test = RelationDataset(ftest,  tokenizer=tokenizer)
    voc_emb = EnEmbedding(spacy_model)
    
    fresult = f"../result/{dataset}/cleanlab_ep10.txt"
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            folder = os.path.join(home, f"model/{dataset}/cleanlab/{arch}/ep10/{k}/")
            models = [
                ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation),
                ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
            ]
            num_neg  = count_neg(train)
            ratio    = fn * 0.1
            cleanlab = CleanLab(models[0], voc_emb, cuda_devices=cuda)
            train, wgt = cleanlab.clean(folder, train, valid, n_fold=n_fold, epochs=epochs, batch=batch, lr=lr)
            new_neg  = count_neg(train)
            ratio    = max(0, ratio - (new_neg / num_neg))

            models[0].reset_net()
            trainer  = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
            trainer.train(folder, train, valid, ratio=ratio, wgt=wgt, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
            f1, p, r = trainer.test(folder, 0, test, batch=batch)

            print(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.flush()
    file.close()
    return

def test_model_nyt():
    cuda = '0'
    dataset = 'nyt'
    spacy_model = 'en_vectors_web_lg'
    archs = ['cnn', 'pcnn']
    max_len = 256
    batch = 256
    n_policy = 3
    n_relation = 53

    home = os.path.expanduser("~")
    ftest = os.path.join(home, f"dataset/{dataset}/test.json")

    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len, bagging=True)
    test        = RelationDataset(ftest,  tokenizer=tokenizer)
    voc_emb     = EnEmbedding(spacy_model)

    fresult = f"../result/{dataset}/cleanlab_test_clean.txt"
    file = open(fresult, 'a')
    for arch in archs:
        for n in range(0, 5):
            folder = os.path.join(home, f"model/{dataset}/cleanlab/{arch}/{n}/")
            print('Testing model: ', n)
            print('Model file in: ', folder)

            models = [
                ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation),
                ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
            ]

            trainer = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
            # f1, p, r = trainer.test(folder, 0, test, batch=batch)
            # file.write(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            # file.flush()
            fpred_folder = os.path.join(home, f"ckip_re/predict/{dataset}_{arch}_cleanlab/")
            assure_folder_exist(fpred_folder)
            fpred = os.path.join(fpred_folder, f"test_{n}.npz")
            targets = np.array([r for r, _, _, _, _ in test.data])
            logits  = trainer.predict(folder, 0, test, batch=batch).cpu().numpy()
            np.savez(fpred, targets=targets, logits=logits)
    file.close()
    return

def train_zhwiki():
    cuda        = '0'
    dataset     = 'zhwiki'
    arch        = 'cnn'
    n_fold      = 4
    n_relation  = 52
    max_len     = 256
    pre_epochs  = 10
    epochs      = 200
    batch       = 256
    lr          = 0.0001
    ratio       = 0.2
    
    home        = os.path.expanduser("~")
    folder      = os.path.join(home, f"model/{dataset}/{arch}/")
    fchar_list  = os.path.join(home, f"model/ckiptagger/embedding_character/token_list.npy")
    fword_list  = os.path.join(home, f"model/ckiptagger/embedding_word/token_list.npy")
    fchar_emb   = os.path.join(home, f"model/ckiptagger/embedding_character/vector_list.npy")
    fword_emb   = os.path.join(home, f"model/ckiptagger/embedding_word/vector_list.npy")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test.json")
    #fresult    = os.path.join(home, f"../result/{dataset}_cont_cleanlab_nr.txt")
    f_prob      = os.path.join(home, f"model/{dataset}/{arch}/prob.npy")
    f_qtable    = os.path.join(home, f"model/{dataset}/{arch}/qtable.npy")
    
    tokenizer   = ZhTokenizer(fchar_list, fword_list, max_len=max_len)
    train       = RelationDataset(ftrain, tokenizer=tokenizer)
    valid       = RelationDataset(fvalid, tokenizer=tokenizer)
    test        = RelationDataset(ftest,  tokenizer=tokenizer)

    voc_emb     = ZhEmbedding(fchar_emb, fword_emb)
    model       = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)

    cleanlab    = CleanLab(model, voc_emb, cuda_devices=cuda)
    #prob_train  = cleanlab.estimate_prob(folder, train, valid, n_fold, epochs, batch, lr)
    #qtable      = cleanlab.estimate_qtable(train, prob_train)

    #with open(f_prob, 'wb')   as file: np.save(file, prob_train)
    #with open(f_qtable, 'wb') as file: np.save(file, qtable)
    models      = [
        model, ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
    ]

    with open(f_prob, 'rb')   as file: prob_train = np.load(file)
    with open(f_qtable, 'rb') as file: qtable     = np.load(file)
    train, wgt  = cleanlab.clean_dataset(train, prob_train, qtable)
    trainer     = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)

    #trainer.train(folder, train, valid, ratio=ratio, wgt=wgt, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
    f1, p, r = trainer.test(folder, 0, test, batch=batch)
    print(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}")

    return

if __name__ == '__main__':
    train_nyt()
    # test_model_nyt()