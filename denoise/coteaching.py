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
from util.embedding import EnEmbedding
from util.tokenizer import EnTokenizer
from util.measure import OldMicroF1
from base.cnn import ModelCNN
from base.rbert import RBert

class CoteachingLoss(nn.Module):
    def __init__(self, wgt=1.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.wgt  = torch.tensor(wgt).cuda()
        return

    def forward(self, ys, target, remember_rate):
        idxes  = [None] * len(ys)
        for i in range(len(ys)):
            ls = self.loss(ys[i], target)
            ls[target!=0] = 0.0 # force remembering positives
            idxes[i] = torch.argsort(ls)

        # num    = int(len(target) * remember_rate)
        num    = int((target!=0).sum() + (target==0).sum() * remember_rate) # remove negatives only
        idxes  = idxes[-1:] + idxes[:-1] # rotate 1 space to right
        losses = [None] * len(ys)
        for i in range(len(ys)):
            idx = idxes[i][:num]
            y   = ys[i][idx]
            t   = target[idx]
            w   = self.wgt.expand(y.size(1))[t]
            losses[i]  = (self.loss(y, t) * w).sum()

        return losses

class CoteachingTrainer:
    def __init__(self, models, voc_emb, cuda_devices='0'):
        # cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        torch.multiprocessing.set_sharing_strategy('file_system')

        # models
        self.voc_emb = nn.DataParallel(voc_emb.cuda())
        self.models  = [nn.DataParallel(model.cuda()) for model in models]
        return

    def train(self, folder, train, valid, ratio=0, wgt=1.0, pre_epochs=10, epochs=100, batch=256, lr=0.001): 
        assure_folder_exist(folder)
        train = DataLoader(train, batch_size=batch, shuffle=True)
        valid = DataLoader(valid, batch_size=batch)
        bests = [-1] * len(self.models)

        loss   = CoteachingLoss(wgt).cuda()
        optims = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models] #, amsgrad=True)

        for e in tqdm(range(epochs)):
            remember_rate = 1 - ratio * min(e/pre_epochs, 1.0)
            scores = self.update_model(train, loss, optims, remember_rate)
            for i, (f1, ls) in enumerate(scores):
                print(f"Ep {e}, model {i}: f1_train = {f1:.4f}, loss = {ls}")

            scores = self.test_model(valid)
            for i, (f1, p, r) in enumerate(scores):
                print(f"Ep {e}, model {i}: f1_valid = {f1:.4f}")

            for i in range(len(scores)):
                if bests[i] < scores[i][0]:
                    bests[i] = scores[i][0]
                    torch.save(self.models[i].state_dict(), os.path.join(folder, f'model_{i}.pt')) 

                print(f"Ep {e}, model {i}: f1_best_valid = {bests[i]:.4f}")

        return np.argmax(bests)

    def test(self, folder, best, test, batch=256):
        test  = DataLoader(test, batch_size=batch)
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(folder, f'model_{i}.pt'))) 

        scores = self.test_model(test)
        print(f"best = {best}")
        for i, (f1, p, r) in enumerate(scores):
            print(f"model {i}: f1_test  = {f1:.4f}, p_test  = {p:.4f}, r_test  = {r:.4f}")

        return scores[best]

    def predict(self, folder, best, test, batch=256):
        data = DataLoader(test, batch_size=batch)
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(folder, f'model_{i}.pt')))

        logits = None
        for idx, (r, _, _, voc, pos) in data:
            out = self.models[best](self.voc_emb(voc.cuda()), pos.cuda())
            out = nn.functional.softmax(out, dim=1).detach()
            if logits is None: logits = out
            else: logits = torch.cat((logits, out), 0)
            
        return logits

    def update_model(self, data, loss, optims, remember_rate):
        scores = [[OldMicroF1(), 0] for model in self.models]

        for model in self.models: model.train()
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            ys  = [model(self.voc_emb(voc.cuda()), pos.cuda()) for model in self.models]
            lss = loss(ys, r, remember_rate)

            for optim, ls in zip(optims, lss):
                optim.zero_grad()
                ls.backward()
                optim.step()
            
            for y, ls, score in zip(ys, lss, scores):
                selected = torch.max(y, dim=1)[1].detach()
                score[0].add(selected, r)
                score[1] += ls.detach().cpu().numpy()
        
        for score in scores: score[0] = score[0].get().cpu().numpy()
        return scores

    def test_model(self, data):
        scores = [[OldMicroF1(), 0, 0] for model in self.models]

        for model in self.models: model.eval()
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            ys  = [model(self.voc_emb(voc.cuda()), pos.cuda()) for model in self.models]

            for y, score in zip(ys, scores):
                selected = torch.max(y, dim=1)[1].detach()
                score[0].add(selected, r)

        for score in scores: 
            score[0], score[1], score[2] = score[0].get(full=True, bagging=True) 
            for i in range(3): score[i] = score[i].cpu().numpy()

        return scores

def train_semeval():
    cuda        = '0'
    dataset     = 'semeval2010'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fns         = range(0, 6)
    n_relation  = 10
    max_len     = 100
    pre_epochs  = 10
    epochs      = 150
    batch       = 256
    lr          = 0.003
    repeat      = 4
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_coteaching_fn.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}_{arch}_coteaching/fn_{fn}/")
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
                #ratio    = train.stat_relation()[0] * (fn * 0.1)
                ratio    = fn * 0.1
                #print(ratio)
                trainer  = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
                trainer.train(folder, train, valid, ratio=ratio, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
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
    fns         = range(6, 10)
    n_relation  = 42
    max_len     = 100
    pre_epochs  = 10
    epochs      = 200
    batch       = 256
    lr          = 0.0003
    repeat      = 5
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_cont_coteaching.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}_cont_{arch}_coteaching/fn_{fn}/")
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
                #ratio    = train.stat_relation()[0] * (fn * 0.1)
                ratio    = fn * 0.1
                #print(ratio)
                trainer  = CoteachingTrainer(models, voc_emb, cuda_devices=cuda)
                trainer.train(folder, train, valid, ratio=ratio, pre_epochs=pre_epochs, epochs=epochs, batch=batch, lr=lr)
                f1, p, r = trainer.test(folder, 0, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return


if __name__ == '__main__':
    train_tacred()