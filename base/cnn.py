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
from util.measure import MicroF1

class ModelCNN(nn.Module):
    def __init__(self, arch='cnn', max_len=100, voc_dim=300, pos_dim=50, 
                 cnn_filters=[2, 3, 4, 5], cnn_dim=230, dropout=0.5, n_class=10):
        super().__init__()

        # architecture parameters
        self.arch           = arch
        self.max_len        = max_len
        self.voc_dim        = voc_dim
        self.pos_dim        = pos_dim
        
        self.cnn_dim        = cnn_dim
        self.cnn_filters    = cnn_filters
        self.n_feature      = self.cnn_dim * len(self.cnn_filters) * (3 if arch == 'pcnn' else 1)
        self.dropout_rate   = dropout
        self.n_class        = n_class

        # network layers
        self.dropout        = nn.Dropout(self.dropout_rate)
        self.create_net()
        return

    def create_net(self):
        self.head_pos_emb   = nn.Embedding(self.max_len*2, self.pos_dim) # *2 for +-len relative pos
        self.tail_pos_emb   = nn.Embedding(self.max_len*2, self.pos_dim)

        self.cnn            = nn.ModuleList( [ nn.Sequential(
                                    nn.Conv1d(self.voc_dim + self.pos_dim*2, self.cnn_dim, f), 
                                    nn.ConstantPad1d((0, f-1), 0)
                              ) for f in self.cnn_filters ] )

        self.dnn            = nn.Sequential(
                                nn.Linear(self.n_feature, self.n_class)
                              )
        return

    def reset_net(self):
        self.create_net()
        return

    def get_dropout_infos(self, batch):
        embed = self.voc_dim + 2 * self.pos_dim
        return self.dropout_rate, [(batch, embed, self.max_len), (batch, self.n_feature)]

    def forward(self, voc, pos, dropouts=None):
        pos_h  = self.head_pos_emb(pos[:, 0, :])
        pos_t  = self.tail_pos_emb(pos[:, 1, :])
        pieces = pos[:, 2:, :].type(voc.type())

        inputs = torch.cat([voc, pos_h, pos_t], dim=2) # shape = batch x time x embed
        inputs*= torch.sum(pieces, dim=1).unsqueeze(-1).expand(-1, -1, inputs.size(2))
        inputs = inputs.transpose(1, 2)  # shape = batch x embed x time
        if dropouts is None: inputs = self.dropout(inputs)
        else:                inputs = inputs * dropouts[0]

        cnns = [] # select cnn architecture, pool and concat

        if self.arch == 'pcnn':
            #masks = [pieces[:, i:i+1, :].expand(-1, self.cnn_dim, -1) for i in range(3)]
            masks = [
                (pieces[:, 0:1, :] + pieces[:, 1:2, :]).expand(-1, self.cnn_dim, -1), 
                (pieces[:, 2:3, :] + pieces[:, 3:4, :]).expand(-1, self.cnn_dim, -1),
                (pieces[:, 4:5, :]).expand(-1, self.cnn_dim, -1),
            ]
            for f in range(len(self.cnn_filters)):
                conv = self.cnn[f](inputs)  #shape = batch x cnn_dim x time
                ps   = [torch.max(conv*mask, dim=2, keepdim=False)[0] for mask in masks]
                pool = torch.cat(ps, dim=1) #shape = batch x (cnn_dim * 3)
                cnns.append(pool)
        else:
            for f in range(len(self.cnn_filters)):
                conv = self.cnn[f](inputs)
                pool = torch.max(conv, dim=2, keepdim=False)[0]
                cnns.append(pool)

        feature = torch.cat(cnns, dim=1)  # shape = batch x n_feature

        if dropouts is None: feature = self.dropout(feature)
        else:                feature = feature * dropouts[1]
            
        out     = self.dnn(feature)
        return out


class Trainer:
    def __init__(self, model, voc_emb, cuda_devices='0'):
        # cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        torch.multiprocessing.set_sharing_strategy('file_system')

        # models
        self.voc_emb  = nn.DataParallel(voc_emb.cuda())
        self.model    = nn.DataParallel(model.cuda())
        return

    def train(self, folder, train, valid, epochs=100, batch=256, lr=0.001):
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

    def predict(self, folder, test, batch=256):
        data = DataLoader(test, batch_size=batch)
        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        logits = None
        
        for idx, (r, _, _, voc, pos) in data:
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            out = nn.functional.softmax(out, dim=1).detach()
            if logits is None: logits = out
            else: logits = torch.cat((logits, out), 0)
            
        return logits
 
    def update_model(self, data, loss, optim, mode=None):
        f1 = MicroF1()
        ls_ep = 0

        self.model.train()
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            ls  = loss(out, r).sum()

            optim.zero_grad()
            ls.backward()
            optim.step()
            
            # selected = torch.max(out, dim=1)[1].detach()
            out = torch.nn.functional.softmax(out, dim=1)
            f1.add(out.detach(), r, h, t)
            ls_ep += ls.detach().cpu().numpy()
        
        f1 = f1.get().cpu().numpy()
        return f1, ls_ep

    def test_model(self, data, mode=None, full=False):
        f1 = MicroF1()
        self.model.eval()
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            out = torch.nn.functional.softmax(out, dim=1).detach()
            f1.add(out, r, h, t)

        if not full: return f1.get().cpu().numpy() 

        f1, p, r = f1.get(full=full, bagging=False) 
        f1, p, r = f1.cpu().numpy(), p.cpu().numpy(), r.cpu().numpy()
        return f1, p, r


def train_semeval():
    cuda        = '0'
    dataset     = 'semeval2010'
    arch        = 'cnn'
    spacy_model = 'en_vectors_web_lg'
    n_relation  = 10
    max_len     = 100
    epochs      = 150
    batch       = 256
    lr          = 0.003
    fn          = 0
    
    home        = os.path.expanduser("~")
    folder      = os.path.join(home, f"model/{dataset}/{arch}/")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train_fn_0.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid_fn_0.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test.json")
    
    nlp         = spacy.load(spacy_model)
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    train       = RelationDataset(ftrain, tokenizer=tokenizer)
    valid       = RelationDataset(fvalid, tokenizer=tokenizer)
    test        = RelationDataset(ftest,  tokenizer=tokenizer)

    voc_emb     = EnEmbedding(spacy_model)
    model       = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
    trainer     = Trainer(model, voc_emb, cuda_devices=cuda)

    trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
    f1, p, r = trainer.test(folder, test, batch=batch)
    print(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}")
    return


def train_tacred():
    cuda        = '1'
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
    repeat      = 4
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_cont_cnn.txt"
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
                model  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)

                trainer  = Trainer(model, voc_emb, cuda_devices=cuda)
                trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
                f1, p, r = trainer.test(folder, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()    
    return


def train_nyt():
    cuda        = '0'
    dataset     = 'nyt'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    n_relation  = 53
    max_len     = 256
    epochs      = 30
    batch       = 256
    lr          = 3e-4
    repeats     = 5
    
    home        = os.path.expanduser("~")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test_clean.json")
    
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len, bagging=True)
    train       = RelationDataset(ftrain, tokenizer=tokenizer)
    valid       = RelationDataset(fvalid, tokenizer=tokenizer)
    test        = RelationDataset(ftest,  tokenizer=tokenizer)
    print('Finish loading datasets.')
    voc_emb     = EnEmbedding(spacy_model)

    fresult = f'../result/{dataset}/base_pcnn.txt'
    file = open(fresult, 'a')
    for arch in archs:
        for r in range(repeats):
            folder      = os.path.join(home, f"model/{dataset}/{arch}/{r}/")
            model       = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
            trainer     = Trainer(model, voc_emb, cuda_devices=cuda)

            print('Training ...')
            trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr=lr)
            print('Testing ...')
            f1, p, r = trainer.test(folder, test, batch=batch)
            print(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}")

            file.write(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
            file.flush()
    file.close()
    return

if __name__ == '__main__':
    train_nyt()