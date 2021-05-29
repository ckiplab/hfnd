#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.data import RelationDataset, FNSimRetriever, assure_folder_exist
from util.embedding import EnEmbedding
from util.tokenizer import EnTokenizer
from util.measure import MicroF1
from base.cnn import ModelCNN

class AgentLoss(nn.Module):
    def __init__(self, window=5):
        super(AgentLoss, self).__init__()
        self.window  = window
        self.rewards = []
    
    def forward(self, policy, selected, rtarget, reward):
        if len(self.rewards)>0:
            reward -= np.average(self.rewards)
        else:
            reward = 0

        idx      = torch.arange(rtarget.size(0))
        log_p    = nn.functional.log_softmax(policy, dim=1)[idx, selected]
        mask_p   = (rtarget==0).float()
        ls       = (log_p * mask_p).sum()
        out      = -reward * ls
        
        return out

    def update(self, reward):
        self.rewards.append(reward)
        
        if len(self.rewards) > self.window:
            self.rewards = self.rewards[1:]
        return


class Trainer:
    def __init__(self, model, agent, voc_emb, cuda_devices='0'):
        # cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        torch.multiprocessing.set_sharing_strategy('file_system')

        # models
        self.voc_emb  = nn.DataParallel(voc_emb.cuda())
        self.model    = nn.DataParallel(model.cuda())
        self.agent    = nn.DataParallel(agent.cuda())
        return

    def test(self, folder, test, batch=256, bagging=False):
        test = DataLoader(test, batch_size=batch)
        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        f1, p, r = self.test_model(test, full=True, bagging=bagging)

        print(f"f1_test  = {f1:.4f}, p_test  = {p:.4f}, r_test  = {r:.4f}")
        return f1, p, r

    def predict(self, folder, test_dataset, batch=256):
        data_loader = DataLoader(test_dataset, batch_size=batch)
        self.model.load_state_dict(torch.load(os.path.join(folder, 'model.pt')))
        logits = None
        
        for idx, (_, _, _, voc, pos) in data_loader:
            out = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            out = nn.functional.softmax(out, dim=1).detach()

            if logits is None:
                logits = out
            else:
                logits = torch.cat((logits, out), 0)
            
        return logits

    def pretrain(self, folder, train, valid, epochs_model=0, epochs_agent=0, batch=128, lr_model=0.001, lr_agent=0.001):
        assure_folder_exist(folder)
        train = DataLoader(train, batch_size=batch, shuffle=True)
        valid = DataLoader(valid, batch_size=batch)

        loss_model  = nn.CrossEntropyLoss(reduction='none').cuda()
        optim_model = torch.optim.Adam(self.model.parameters(), lr=lr_model)#, amsgrad=True)
        loss_agent  = nn.CrossEntropyLoss(reduction='none').cuda()
        optim_agent = torch.optim.Adam(self.agent.parameters(), lr=lr_agent)#, amsgrad=True)

        for e in tqdm(range(epochs_model)):
            f1 = self.update_model(train, loss_model, optim_model)
            print(f"Ep {e}: f1_train = {f1:.4f}")

            f1 = self.test_model(valid)
            print(f"Ep {e}: f1_valid = {f1:.4f}")

        for e in tqdm(range(epochs_agent)):
            f1, ls = self.supervise_agent(train, loss_agent, optim_agent)
            print(f"Ep {e}: f1_agent = {f1:.4f}, loss = {ls:.4f}")

            f1 = self.test_model(valid)
            print(f"Ep {e}: f1_valid = {f1:.4f}")
        
        # torch.save(self.model.state_dict(), os.path.join(folder, 'model.pt'))
        # torch.save(self.agent.state_dict(), os.path.join(folder, 'agent.pt'))
        return

    def train(self, folder, train, valid, epochs=100, batch=128, lr_model=0.001, lr_agent=0.001, lr_decay=0.97):
        assure_folder_exist(folder)
        train = DataLoader(train, batch_size=batch, shuffle=True)
        valid = DataLoader(valid, batch_size=batch)

        loss_model  = nn.CrossEntropyLoss(reduction='none').cuda()
        optim_model = torch.optim.Adam(self.model.parameters(), lr=lr_model)#, amsgrad=True)
        loss_agent  = AgentLoss().cuda()
        optim_agent = torch.optim.Adam(self.agent.parameters(), lr=lr_agent)#, amsgrad=True)

        for e in tqdm(range(epochs)):
            # revise and update model
            s = torch.random.get_rng_state()
            rev_rel_train, rev_pol_train = self.revise(train, batch, train=True)
            f1 = self.update_model(train, loss_model, optim_model, rev_rel_train, rev_pol_train)
            print(f"Ep {e}: f1_train = {f1:.4f}")

            # revise and get reward from valid
            rwd = 0
            for i in range(5):
                rev_rel_valid, rev_pol_valid = self.revise(valid, batch, train=False)
                f1 = self.test_model(valid, rev_rel_valid, rev_pol_valid)
                rwd += f1
            rwd /= 5
            print(f"Ep {e}: f1_valid = {rwd:.4f}")

            # update agent
            torch.random.set_rng_state(s) # enforce every random value identical to that in revise
            ls = self.update_agent(train, loss_agent, optim_agent, rev_rel_train, rev_pol_train, rwd)
            loss_agent.update(rwd)

            for pg in optim_model.param_groups: pg['lr'] *= lr_decay
            for pg in optim_agent.param_groups: pg['lr'] *= lr_decay

            # save model
            torch.save(self.model.state_dict(), os.path.join(folder, 'model.pt'))
            torch.save(self.agent.state_dict(), os.path.join(folder, 'agent.pt'))

            #if (e % 10 == 9):
                #self.stat_policy(train_datset, batch=batch)

        return

    def revise(self, data, batch, train=True):
        rev_rel = torch.zeros(len(data)*batch, dtype=torch.long)
        rev_pol = torch.zeros(len(data)*batch, dtype=torch.long)
        dropout0, dropout1 = torch.ones(1), torch.ones(1)
        self.model.eval()
        self.agent.eval()
        if train: self.agent.train()            

        for idx, (r, _, _, voc, pos) in data:
            p, dropout_sizes = self.model.module.get_dropout_infos(idx.size()[0])
            dropout0 = torch.bernoulli(p * torch.ones(dropout_sizes[0]))
            dropout1 = torch.bernoulli(p * torch.ones(dropout_sizes[1]))
            dropouts = [dropout0.cuda(), dropout1.cuda()]

            r   = r.cuda()
            voc = self.voc_emb(voc.cuda())
            pos = pos.cuda()
            rel = self.model(voc, pos, dropouts=dropouts)
            pol = self.agent(voc, pos, dropouts=dropouts)
            rel = nn.functional.softmax(rel, dim=1)
            pol = nn.functional.softmax(pol, dim=1)
            rel[:, 0]  = 0

            sel_rel = torch.argmax(rel, dim=1)
            sel_pol = torch.multinomial(pol, 1).view((-1))

            mask_rel = ~((r==0) & (sel_pol==2))
            sel_rel[mask_rel]  = r[mask_rel]
            sel_pol[r != 0]    = -1
            rev_rel[idx.cpu()] = sel_rel.detach().cpu()
            rev_pol[idx.cpu()] = sel_pol.detach().cpu()

        return rev_rel, rev_pol

    def update_model(self, data, loss, optim, rev_rel=None, rev_pol=None):
        f1 = MicroF1()
        self.model.train()
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            wgt = 1.0

            if rev_rel is not None: r = rev_rel[idx.cpu()].cuda()
            if rev_pol is not None: 
                sel_pol = rev_pol[idx.cpu()].cuda()
                wgt = (sel_pol!=1).unsqueeze(-1).float()

            rel = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            ls  = (loss(rel, r) * wgt).sum()

            optim.zero_grad()
            ls.backward()
            optim.step()
            
            rel = torch.nn.functional.softmax(rel, dim=1).detach()          
            f1.add(rel, r, h, t, wgt)

        f1 = f1.get().cpu().numpy() 
        return f1

    def test_model(self, data, rev_rel=None, rev_pol=None, full=False, bagging=False):
        f1 = MicroF1()
        self.model.eval()
        for idx, (r, h, t, voc, pos) in data:
            wgt = 1.0
            if rev_rel is not None: r = rev_rel[idx.cpu()]
            if rev_pol is not None: 
                sel_pol = rev_pol[idx.cpu()].cuda()
                wgt = (sel_pol!=1).unsqueeze(-1).float()

            rel = self.model(self.voc_emb(voc.cuda()), pos.cuda())
            rel = torch.nn.functional.softmax(rel, dim=1)
            r   = r.cuda()
            f1.add(rel.detach(), r, h, t, wgt)

        if not full: return f1.get().cpu().numpy() 

        f1, p, r = f1.get(full=full, bagging=bagging) 
        f1, p, r = f1.cpu().numpy(), p.cpu().numpy(), r.cpu().numpy()
        return f1, p, r

    def update_agent(self, data, loss, optim, rev_rel, rev_pol, rwd):
        ls_ep = 0
        self.agent.train()
        for idx, (r, _, _, voc, pos) in data:
            p, dropout_sizes = self.model.module.get_dropout_infos(idx.size()[0])
            dropout0 = torch.bernoulli(p * torch.ones(dropout_sizes[0]))
            dropout1 = torch.bernoulli(p * torch.ones(dropout_sizes[1]))
            dropouts = [dropout0.cuda(), dropout1.cuda()]

            r   = r.cuda()
            voc = self.voc_emb(voc.cuda())
            pos = pos.cuda()
            rel = self.model(voc, pos, dropouts=dropouts)
            pol = self.agent(voc, pos, dropouts=dropouts)

            # because the model is updated in previous batches, 
            # using the sample result in revise is necessary even if manual seed is used
            sel_pol = rev_pol[idx.cpu()].cuda().detach()
            ls = loss(pol, sel_pol, r, rwd)

            optim.zero_grad()
            ls.backward()
            optim.step()

            ls_ep += ls.detach().cpu().numpy()

            # parallalization of function call using random in revise, for identical random results
            pol = nn.functional.softmax(pol, dim=1)
            torch.multinomial(pol, 1).view((-1))

        return ls_ep

    def supervise_agent(self, data, loss, optim):
        f1 = MicroF1()
        self.model.eval()
        self.agent.train()
        ls_ep = 0
        for idx, (r, h, t, voc, pos) in data:
            r   = r.cuda()
            voc = self.voc_emb(voc.cuda())
            pos = pos.cuda()
            rel = self.model(voc, pos)
            pol = self.agent(voc, pos)
            sel_rel = torch.argmax(rel, dim=1)
            revise  = (r!=0) & (sel_rel==r)
            keep    = (r==0) & (sel_rel==r)

            sup = torch.ones(r.size()).type(r.type())
            sup[revise] = 2
            sup[keep] = 0
            ls  = (loss(pol, sup)).sum()

            optim.zero_grad()
            ls.backward()
            optim.step()
            
            ls_ep  += ls.detach().cpu().numpy()
            pol = torch.nn.functional.softmax(pol, dim=1).detach()
            f1.add(pol, sup, h, t)

        f1 = f1.get().cpu().numpy()
        return f1, ls_ep

    def stat_policy(self, folder, dataset, batch=256):
        data = DataLoader(dataset, batch_size=batch)
        self.agent.load_state_dict(torch.load(os.path.join(folder, 'agent.pt')))

        # stat
        self.agent.eval()
        stat_stochastic = torch.zeros((2, 3))
        stat_determined = torch.zeros((2, 3))
        for idx, (rd, _, _, voc, pos) in data:
            rd  = rd.cuda()
            voc = self.voc_emb(voc.cuda())
            pos = pos.cuda()
            pol = self.agent(voc, pos)

            prob = nn.functional.softmax(pol.detach().cpu(), dim=1)
            stat_stochastic[0] += torch.sum(prob, 0)
            for i in range(3):
                stat_determined[0][i] += (torch.max(prob, dim=1)[1] == i).sum()
                                
        stat_stochastic[0] /= stat_stochastic[0].sum()
        stat_stochastic[1] /= stat_stochastic[1].sum()
        stat_determined[0] /= stat_determined[0].sum()
        stat_determined[1] /= stat_determined[1].sum()

        print("stat_stochastic:")
        print(stat_stochastic)
        print("stat_determined:")
        print(stat_determined)

        return stat_stochastic, stat_determined


def train_semeval():
    cuda        = '3'
    dataset     = 'semeval2010-old'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fns         = range(6)
    n_relation  = 10
    n_policy    = 3
    max_len     = 100
    pre_model   = 5
    pre_agent   = 20
    epochs      = 150
    batch       = 256
    lr_pre      = 3e-3
    lr_model    = 3e-3
    lr_agent    = 3e-5
    lr_decay    = 0.95
    repeat      = 1
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}/aaai_rl.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len, bagging=True)
    retriever   = FNSimRetriever()
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}/aaai-old/{fn}")
                assure_folder_exist(folder)
                print(folder)
                ftrain = os.path.join(home, f"dataset/{dataset}/fn/train_{fn}.json")
                fvalid = os.path.join(home, f"dataset/{dataset}/fn/valid_{fn}.json")
                ftest  = os.path.join(home, f"dataset/{dataset}/fn/test.json")
                train  = RelationDataset(ftrain, tokenizer=tokenizer, retriever=retriever)
                valid  = RelationDataset(fvalid, tokenizer=tokenizer, retriever=retriever)
                test   = RelationDataset(ftest,  tokenizer=tokenizer, retriever=retriever)
                model  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
                agent  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_policy)

                trainer = Trainer(model, agent, voc_emb, cuda_devices=cuda)
                trainer.pretrain(folder, train, valid, epochs_model=pre_model, epochs_agent=pre_agent, batch=batch, lr_model=lr_pre, lr_agent=lr_pre)
                #trainer.stat_policy(train, batch=batch)

                trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr_model=lr_model, lr_agent=lr_agent, lr_decay=lr_decay)
                f1, p, r = trainer.test(folder, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return

def train_tacred():
    cuda        = '1'
    dataset     = 'tacred'
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fns         = range(0, 6)
    n_relation  = 42
    n_policy    = 3
    max_len     = 100
    pre_model   = 5
    pre_agent   = 20
    epochs      = 200
    batch       = 256
    lr_pre      = 3e-4
    lr_model    = 3e-4
    lr_agent    = 3e-6
    lr_decay    = 0.97
    repeat      = 5
    
    home        = os.path.expanduser("~")
    fresult     = f"../result/{dataset}_rl.txt"
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    retriever   = FNSimRetriever()
    voc_emb     = EnEmbedding(spacy_model)
    
    file = open(fresult, "a")
    for k in range(repeat):
        for arch in archs:
            for fn in fns:
                folder = os.path.join(home, f"model/{dataset}_{arch}_rl/fn_{fn}/")
                print(folder)
                ftrain = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
                fvalid = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
                ftest  = os.path.join(home, f"dataset/{dataset}/test.json")
                train  = RelationDataset(ftrain, tokenizer=tokenizer, retriever=retriever)
                valid  = RelationDataset(fvalid, tokenizer=tokenizer, retriever=retriever)
                test   = RelationDataset(ftest,  tokenizer=tokenizer, retriever=retriever)
                model  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
                agent  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_policy)

                trainer  = Trainer(model, agent, voc_emb, cuda_devices=cuda)
                trainer.pretrain(folder, train, valid, epochs_model=pre_model, epochs_agent=pre_agent, batch=batch, lr_model=lr_pre, lr_agent=lr_pre)
                trainer.stat_policy(folder, train, batch=batch)
                trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr_model=lr_model, lr_agent=lr_agent, lr_decay=lr_decay)
                trainer.stat_policy(folder, train, batch=batch)
                f1, p, r = trainer.test(folder, test, batch=batch)
                file.write(f"arch = {arch}, fn = {fn}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
                file.flush()

    file.close()
    return


def train_nyt():
    cuda        = '1'
    dataset     = 'nyt'
    arch        = 'cnn'
    spacy_model = 'en_vectors_web_lg'
    n_policy    = 3
    n_relation  = 53
    max_len     = 256
    pre_model   = 30
    pre_agent   = 20
    epochs      = 20
    batch       = 256
    lr_pre      = 3e-4
    lr_model    = 3e-4
    lr_agent    = 3e-6
    lr_decay    = 0.97
    repeat      = 1
    
    home        = os.path.expanduser("~")
    folder      = os.path.join(home, f"model/{dataset}/{arch}_rl/ep20/0/")
    ftrain      = os.path.join(home, f"dataset/{dataset}/train.json")
    fvalid      = os.path.join(home, f"dataset/{dataset}/valid.json")
    ftest       = os.path.join(home, f"dataset/{dataset}/test_clean.json")
    
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len, bagging=True)
    train       = RelationDataset(ftrain, tokenizer=tokenizer)
    valid       = RelationDataset(fvalid, tokenizer=tokenizer)
    test        = RelationDataset(ftest,  tokenizer=tokenizer)
    voc_emb     = EnEmbedding(spacy_model)

    print(folder)
    fresult = f'../result/{dataset}/rl_cnn_ep20.txt'
    file = open(fresult, 'a')
    for r in range(repeat):
        model  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
        agent  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_policy)

        trainer  = Trainer(model, agent, voc_emb, cuda_devices=cuda)
        trainer.pretrain(folder, train, valid, epochs_model=pre_model, epochs_agent=pre_agent, batch=batch, lr_model=lr_pre, lr_agent=lr_pre)
        #trainer.stat_policy(train, batch=batch)
        trainer.train(folder, train, valid, epochs=epochs, batch=batch, lr_model=lr_model, lr_agent=lr_agent, lr_decay=lr_decay)
        f1, p, r = trainer.test(folder, test, batch=batch)
        print(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}")
        file.write(f"arch = {arch}, f1_test = {f1:.4f}, p_test = {p:.4f}, r_test = {r:.4f}\n")
        file.flush()
    file.close()
    return


def model_size():
    cuda        = '0'
    datasets    = ['semeval2010', 'tacred']
    archs       = ['cnn', 'pcnn']
    spacy_model = 'en_vectors_web_lg'
    fn          = 0
    n_relations = [10, 42]
    n_policy    = 3
    max_len     = 100
    
    home        = os.path.expanduser("~")
    tokenizer   = EnTokenizer(spacy_model=spacy_model, max_len=max_len)
    retriever   = FNSimRetriever()
    voc_emb     = EnEmbedding(spacy_model)
    
    for dataset, n_relation in zip(datasets, n_relations):
        for arch in archs:
            ftrain = os.path.join(home, f"dataset/{dataset}/train_fn_{fn}.json")
            fvalid = os.path.join(home, f"dataset/{dataset}/valid_fn_{fn}.json")
            ftest  = os.path.join(home, f"dataset/{dataset}/test.json")
            train  = RelationDataset(ftrain, tokenizer=tokenizer, retriever=retriever)
            valid  = RelationDataset(fvalid, tokenizer=tokenizer, retriever=retriever)
            test   = RelationDataset(ftest,  tokenizer=tokenizer, retriever=retriever)
            model  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_relation)
            agent  = ModelCNN(arch=arch, max_len=max_len, voc_dim=voc_emb.voc_dim, n_class=n_policy)

            model_num_params = sum([param.nelement() for param in model.parameters()])
            agent_num_params = sum([param.nelement() for param in agent.parameters()])
            print(f"len(params) with {arch} for {dataset}:")
            print(f"model: {model_num_params}")
            print(f"agent: {agent_num_params}")

if __name__ == '__main__':
    #train_nyt()
    train_semeval()
