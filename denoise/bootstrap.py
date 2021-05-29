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
from base.cnn import ModelCNN
from base.rbert import RBert

class BootstrapTrainer:
    def __init__(self, model, voc_emb, cuda_devices='0'):
        # cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        torch.multiprocessing.set_sharing_strategy('file_system')

        # models
        self.voc_emb = nn.DataParallel(voc_emb.cuda())
        self.model   = nn.DataParallel(model.cuda())
        return

    def train(self, folder, train, valid, epochs=100, batch=256, lr=0.001):
        train_raw, train_clean = self.split_raw(folder, train)
        valid_raw, valid_clean = self.split_raw(folder, valid)
        #return

        self.train_once(folder, train_clean, valid_clean, epochs=epochs, batch=batch, lr=lr)
        #self.train_once(folder, train_clean, valid, epochs=epochs, batch=batch, lr=lr)
        train = self.merge_raw(folder, train_clean, train_raw, batch=batch)
        valid = self.merge_raw(folder, valid_clean, valid_raw, batch=batch)

        model = self.model.module
        model.reset_net()
        self.model = nn.DataParallel(model.cuda())
        self.train_once(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
        return

    def train_once(self, folder, train, valid, epochs=100, batch=256, lr=0.001):
        assure_folder_exist(folder)
        train = DataLoader(train, batch_size=batch, shuffle=True)
        valid = DataLoader(valid, batch_size=batch)
        best  = -1

        loss_model  = nn.CrossEntropyLoss(reduction='none').cuda()
        optim_model = torch.optim.Adam(self.model.parameters(), lr=lr)#, amsgrad=True)

        for e in tqdm(range(epochs)):
            f1, ls = self.update_model(train, loss_model, optim_model)
            print(f"Ep {e}: f1_train = {f1:.4f}, loss = {ls}")

            f1 = self.test_model(valid)
            print(f"Ep {e}: f1_valid = {f1:.4f}")

            if best < f1:
                best = f1
                torch.save(self.model.state_dict(), os.path.join(folder, 'model.pt'))            
            print("Ep {}: f1_best_valid = {:.4f}".format(e, best))

        return 

    def test(self, folder, test, batch=256):
        test  = DataLoader(test, batch_size=batch)
        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        f1, p, r = self.test_model(test, full=True)

        print(f"f1_test  = {f1:.4f}, p_test  = {p:.4f}, r_test  = {r:.4f}")
        return f1, p, r

    """
    def split_raw(self, folder, dataset, ratio, batch=256):
        data = DataLoader(dataset, batch_size=batch)
        loss = nn.CrossEntropyLoss(reduction='none').cuda()

        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        self.model.eval()

        ep2loss = {}
        for idx, (r, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            ls  = loss(out, r)
            
            for i in range(idx.size(0)):
                if r[i] != 0: continue
                ep = dataset.info[idx[i]]['ep']
                if ep not in ep2loss: ep2loss[ep] = []
                ep2loss[ep].append(ls[i])

        raw_num = int(len(ep2loss) * ratio)
        raw_eps = sorted([(np.mean(ls), ep) for ep, ls in ep2loss.items()], reverse=True)[:raw_num]
        raw, clean = dataset.split(parts=[raw_eps])
        return raw, clean
    """

    def split_raw(self, folder, dataset):
        pos_es = set()
        for i in range(len(dataset)):
            inst = dataset.data[i]
            info = dataset.info[i]

            if inst[0] == 0: continue
            for e in info['ep']: pos_es.add(e)

        n_raw, n_neg = 0, 0
        raw_eps = set()
        for i in range(len(dataset)):
            inst = dataset.data[i]
            info = dataset.info[i]
            ep   = info['ep']

            if inst[0] != 0: continue
            if ep[0] not in pos_es and ep[1] not in pos_es: 
                raw_eps.add(ep)
                n_raw += 1
            else: n_neg += 1

        print(f"n_raw={n_raw}, n_neg={n_neg}")
        raw, clean = dataset.split(parts=[raw_eps])
        return raw, clean

    def merge_raw(self, folder, dataset, raw, batch=256):
        data = DataLoader(raw, batch_size=batch)
        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        self.model.eval()

        ep2prob = {}
        for idx, (r, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            out = out.detach().cpu().numpy()
            
            for i in range(idx.size(0)):
                if r[i] != 0: continue
                ep = raw.info[idx[i]]['ep']
                if ep not in ep2prob: ep2prob[ep] = []
                ep2prob[ep].append(out[i])

        dataset = RelationDataset().merge([dataset])
        for ep in ep2prob:
            probs = np.mean(ep2prob[ep], axis=0)
            r_revised = np.argmax(probs)
            if r_revised == 0: continue

            for idx in raw.bags[ep]:
                inst = raw.data[idx]
                info = raw.info[idx]
                revised_inst = list(inst)
                revised_inst[0] = r_revised
                revised_inst = tuple(revised_inst)
                dataset.insert(revised_inst, info)

        return dataset

    def update_model(self, data, loss, optim, mode=None):
        f1 = MicroF1()
        ls_ep = 0

        self.model.train()
        for idx, (r, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            ls  = loss(out, r).sum()

            optim.zero_grad()
            ls.backward()
            optim.step()
            
            selected = torch.max(out, dim=1)[1].detach()
            f1.add(selected, r)
            ls_ep += ls.detach().cpu().numpy()
        
        f1 = f1.get().cpu().numpy()
        return f1, ls_ep

    def test_model(self, data, mode=None, full=False):
        f1 = MicroF1()
        self.model.eval()
        for idx, (r, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            selected = torch.max(out, dim=1)[1].detach()
            f1.add(selected, r)

        if not full: return f1.get().cpu().numpy() 

        f1, p, r = f1.get(full=full) 
        f1, p, r = f1.cpu().numpy(), p.cpu().numpy(), r.cpu().numpy()
        return f1, p, r


def train_semeval():
    cuda        = '0'
    dataset     = 'semeval2010'
    arch        = 'cnn'
    spacy_model = 'en_vectors_web_lg'
    
    fns         = range(0, 6)
    n_relation  = 10
    max_len     = 100
    epochs      = 150
    batch       = 256
    lr          = 3e-3
    repeat      = 5
    
    home        = os.path.expanduser("~")
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    fresult     = f"../result/{dataset}_{arch}_bootstrap_.txt" 

    file = open(fresult, "a")
    for k in range(repeat):
        for fn in fns:
            folder      = os.path.join(home, f"model/{dataset}_{arch}_bootstrap/fn_{fn}/")
            ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
            fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
            ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

            train       = RelationDataset(ftrain, tokenizer=tokenizer)
            valid       = RelationDataset(fvalid, tokenizer=tokenizer)
            test        = RelationDataset(ftest,  tokenizer=tokenizer)

            voc_emb     = EnEmbedding(spacy_model)
            model       = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
            trainer     = BootstrapTrainer(model, voc_emb, cuda_devices=cuda)

            trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
            f1, p, r = trainer.test(folder, test, batch=batch)
            file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.flush()

    file.close()
    return


def train_tacred():
    cuda        = '0'
    dataset     = 'tacred'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    
    fns         = range(6, 10)
    n_relation  = 42
    max_len     = 100
    epochs      = 200
    batch       = 256
    lr          = 3e-4
    repeat      = 5
    
    home        = os.path.expanduser("~")
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    fresult     = f"../result/{dataset}_cont_bootstrap.txt" 

    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder      = os.path.join(home, f"model/{dataset}_{arch}_bootstrap/fn_{fn}/")
                ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
                fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
                ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

                train       = RelationDataset(ftrain, tokenizer=tokenizer)
                valid       = RelationDataset(fvalid, tokenizer=tokenizer)
                test        = RelationDataset(ftest,  tokenizer=tokenizer)

                voc_emb     = EnEmbedding(spacy_model)
                model       = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
                trainer     = BootstrapTrainer(model, voc_emb, cuda_devices=cuda)

                trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
                f1, p, r = trainer.test(folder, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return

def train_semeval_rbert():
    cuda        = '2'
    dataset     = 'semeval2010'
    arch        = 'rbert'
    bert_model  = 'bert-base-uncased'
    
    fns         = range(0, 6)
    n_relation  = 10
    max_len     = 128
    epochs      = 10
    batch       = 16
    lr          = 2e-5
    repeat      = 5
    
    home        = os.path.expanduser("~")
    tokenizer   = BertEnTokenizer(bert_model=bert_model, max_len=max_len)
    fresult     = f"../result/{dataset}_{arch}_bootstrap.txt" 

    file = open(fresult, "a")
    for k in range(repeat):
        for fn in fns:
            folder      = os.path.join(home, f"model/{dataset}_{arch}_bootstrap/fn_{fn}/")
            ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
            fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
            ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

            train       = RelationDataset(ftrain, tokenizer=tokenizer)
            valid       = RelationDataset(fvalid, tokenizer=tokenizer)
            test        = RelationDataset(ftest,  tokenizer=tokenizer)

            voc_emb     = BertEnVocFeatures()
            model       = RBert(bert_model=bert_model, n_class=n_relation)
            trainer     = BootstrapTrainer(model, voc_emb, cuda_devices=cuda)

            trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
            f1, p, r = trainer.test(folder, test, batch=batch)
            file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.flush()

    file.close()
    return


def train_tacred_rbert():
    cuda        = '3'
    dataset     = 'tacred'
    arch        = 'rbert'
    bert_model  = 'bert-base-uncased'
    
    fns         = range(0, 6)
    n_relation  = 42
    max_len     = 384
    epochs      = 10
    batch       = 6
    lr          = 1e-5
    repeat      = 5
    
    home        = os.path.expanduser("~")
    tokenizer   = BertEnTokenizer(bert_model=bert_model, max_len=max_len)
    fresult     = f"../result/{dataset}_{arch}_bootstrap.txt" 

    file = open(fresult, "a")
    for k in range(repeat):
        for fn in fns:
            folder      = os.path.join(home, f"model/{dataset}_{arch}_bootstrap/fn_{fn}/")
            ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
            fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
            ftest       = os.path.join(home, f"dataset/{dataset}/test.json")

            train       = RelationDataset(ftrain, tokenizer=tokenizer)
            valid       = RelationDataset(fvalid, tokenizer=tokenizer)
            test        = RelationDataset(ftest,  tokenizer=tokenizer)

            voc_emb     = BertEnVocFeatures()
            model       = RBert(bert_model=bert_model, n_class=n_relation)
            trainer     = BootstrapTrainer(model, voc_emb, cuda_devices=cuda)

            trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
            f1, p, r = trainer.test(folder, test, batch=batch)
            file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.flush()

    file.close()
    return


if __name__ == '__main__':
    train_tacred()