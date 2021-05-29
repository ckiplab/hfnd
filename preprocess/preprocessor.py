#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
import json
import random
import spacy
from tqdm import tqdm
from util.data import RelationDataset, FNSimRetriever, assure_folder_exist

class Preprocessor:
    def __init__(self, rn='NA'):
        self.rn    = rn
        self.rels  = {}
        self.ents  = {}
        self.train = RelationDataset()
        self.valid = RelationDataset()
        self.test  = RelationDataset()
        return

    def load(self):
        return

    def dump(self, ftrain=None, fvalid=None, ftest=None):
        #assure_folder_exist(folder)

        # relations
        #fname = os.path.join(folder, "rel2idx.json")
        #print("dump:", fname)
        #with open(fname, 'w') as file: json.dump(self.rels, file, indent=0)
        
        # entities
        #fname = os.path.join(folder, "ent2idx.json")
        #print("dump:", fname)
        #with open(fname, 'w') as file: json.dump(self.ents, file, indent=0)

        # data
        #datasets = [self.train, self.valid, self.test]
        #for dataset, name in zip(datasets, names):
        #    fname = os.path.join(folder, name + '.json')
        #    print("dump:", fname)
        #    dataset.dump(fname)

        datasets = [self.train, self.valid, self.test]
        fnames   = [ftrain, fvalid, ftest]
        for dataset, fname in zip(datasets, fnames):
            if fname is not None:
                print("dump:", fname)
                dataset.dump(fname)
        return


class FNPreprocessor(Preprocessor):
    def __init__(self, rn='NA'):
        super().__init__(rn=rn)
        self.retriever = FNSimRetriever()
        return

    def mask(self, fn_ratio):
        print(fn_ratio)
        self.train = self.mask_dataset(self.train, fn_ratio)
        self.valid = self.mask_dataset(self.valid, fn_ratio)
        return

    def mask_dataset(self, dataset, fn_ratio):
        tp_eps = [] # true positive, no false positive
        fn_eps = [] # false negative
        tn_eps = [] # true negative
        for inst in dataset.data:
            relation = inst['relation']
            rd = relation['id']
            rt = relation['true_id']
            ep = inst['head']['id'], inst['tail']['id']

            if   rt == 0: tn_eps.append(ep)
            elif rd == 0: fn_eps.append(ep)
            else:         tp_eps.append(ep)

        current_ratio = len(fn_eps) / (len(tp_eps) + len(fn_eps))
        if current_ratio > fn_ratio: return

        num = int((fn_ratio - current_ratio) * (len(tp_eps) + len(fn_eps)))
        eps = random.sample(tp_eps, num)
        new_fn, dataset = dataset.split(parts=[eps])
        for inst in new_fn.data:
            inst['relation']['true'] = self.rn
            inst['relation']['true_id'] = 0
            dataset.insert(inst, retriever=self.retriever)
        return dataset


class SemEvalPreprocessor(FNPreprocessor):
    def __init__(self, rn='Other', spacy_model='en_vectors_web_lg'):
        super().__init__(rn=rn)
        self.triples = set()
        self.nlp = spacy.load(spacy_model) 
        return   

    def load_from_processed(self, ftrain, fvalid, ftest):
        self.train.load(ftrain, retriever=self.retriever)
        self.valid.load(fvalid, retriever=self.retriever)
        self.test.load(ftest, retriever=self.retriever)
        return

    def load(self, ftrain, ftest, valid_ratio=0.1):
        self.update_dict(ftest)
        self.update_dict(ftrain)

        # the purpose of following loading order is 
        # preventing training data overlaps testing data
        self.load_dataset(self.test, ftest)
        self.load_dataset(self.train, ftrain)
        self.valid, self.train = self.train.split(ratios=[valid_ratio])
        return
    
    def load_dataset(self, dataset, fname):
        triples = set()
        file = open(fname, 'r')
        line = file.readline()
        while line:
            e1, e2, toks  = self.parse_sentence(line)
            line = file.readline()
            r, head, tail = self.parse_relation(line, e1, e2)
            
            r['id']       = self.rels[r['name']]
            r['true_id']  = self.rels[r['name']]
            head['id']    = self.ents[head['name']]
            tail['id']    = self.ents[tail['name']]

            triple = (r['id'], head['id'], tail['id'])
            if triple in self.triples: continue 
            triples.add(triple)

            inst = {
                'id': len(dataset),
                'relation': r,
                'head': head,
                'tail': tail,
                'toks': toks
            }
            dataset.insert(inst, retriever=self.retriever)

            while line != "\n":
                line = file.readline()
            line = file.readline()

        file.close()
        self.triples.update(triples)
        return

    def update_dict(self, fname):
        rels = set()
        ents = set()
        file = open(fname, 'r')
        line = file.readline()
        while line:
            e1, e2, toks = self.parse_sentence(line)
            line = file.readline()
            r, head, tail = self.parse_relation(line, e1, e2)

            rels.add(r['name'])
            ents.update([head['name'], tail['name']])

            while line != "\n":
                line = file.readline()
            line = file.readline()
        file.close()
        
        rels = list(rels)
        rels.remove(self.rn)
        rels = [self.rn] + sorted(rels)
        ents = sorted(list(ents))
        for i, r in enumerate(rels): self.rels[r] = i
        for i, e in enumerate(ents): self.ents[e] = i

        return

    def parse_sentence(self, line):
        idx, sentence = line.split('\t')
        sentence = sentence[1:-2] # strip " and \n
        delims = ['<e1>', '</e1>', '<e2>', '</e2>']
        for delim in delims:
            sentence = sentence.replace(delim, '<<>>')
        pieces = sentence.split('<<>>')
        pieces = [[tok.text for tok in self.nlp(piece.strip())] for piece in pieces]
        assert (len(pieces) == 5)


        e1_name = ' '.join(pieces[1]).lower() #pieces[1]
        e2_name = ' '.join(pieces[3]).lower() #pieces[3]
        inds    = []
        toks    = []
        for piece in pieces:
            inds.append(len(toks))
            toks += piece

        e1 = {
            'name' : e1_name,
            'start': inds[1],
            'end'  : inds[2] - 1
        }
        e2 = {
            'name' : e2_name,
            'start': inds[3],
            'end'  : inds[4] - 1
        }
        return e1, e2, toks

    def parse_relation(self, line, e1, e2):
        line = line[:-1]
        delims = ['(', ',', ')']
        for delim in delims:
            line = line.replace(delim, ',')
        pieces = line.split(',')
        assert (len(pieces) == 4 or len(pieces) == 1)
        swap = False
        if len(pieces) == 4:
            assert (pieces[1] == 'e1' and pieces[2] == 'e2') or (pieces[1] == 'e2' and pieces[2] == 'e1')
            if pieces[1] == 'e2' and pieces[2] == 'e1': swap = True
            
        relation = {
            'name': pieces[0],
            'true': pieces[0]
        }
        head, tail = e1, e2
        if swap: head, tail = e2, e1
        
        return relation, head, tail


class TacredPreprocessor(FNPreprocessor):
    def __init__(self, rn='no_relation'):
        super().__init__(rn=rn)
        self.triples = set()
        return   

    def load_from_processed(self, ftrain, fvalid, ftest):
        self.train.load(ftrain, retriever=self.retriever)
        self.valid.load(fvalid, retriever=self.retriever)
        self.test.load(ftest, retriever=self.retriever)
        return

    def load(self, ftrain, fvalid, ftest):
        self.update_dict(ftest)
        self.update_dict(ftrain)
        self.update_dict(fvalid)
        
        # the purpose of following loading order is 
        # preventing training data overlaps testing data
        self.load_dataset(self.test, ftest)
        self.load_dataset(self.train, ftrain) # not overlap to testing data
        self.load_dataset(self.valid, fvalid) # not overlap to training and testing data
        return

    def load_dataset(self, dataset, fname):
        print("load:", fname)
        file = open(fname)
        data = json.load(file)
        file.close()

        triples = set()
        for d in data:
            r_name    = d['relation']
            toks      = d['token']
            head_name = ' '.join(toks[d['subj_start']:d['subj_end']+1])
            tail_name = ' '.join(toks[d['obj_start']:d['obj_end']+1])

            r_id = self.rels[r_name]
            h_id = self.ents[head_name]
            t_id = self.ents[tail_name]

            triple = (r_id, h_id, t_id)
            if triple in self.triples: continue 
            triples.add(triple)

            inst = {
                "id": len(dataset),
                "relation" : {
                    "name" : r_name,
                    'id'   : r_id,
                    "true" : r_name,
                    "true_id": r_id
                },
                "head": {
                    "name" : head_name,
                    "id"   : h_id,
                    "start": d['subj_start'],
                    "end"  : d['subj_end']
                },
                "tail": {
                    "name" : tail_name,
                    "id"   : t_id,
                    "start": d['obj_start'],
                    "end"  : d['obj_end']
                },
                "toks"     : toks
            }
            dataset.insert(inst, retriever=self.retriever)
        self.triples.update(triples)
        return  

    def update_dict(self, fname):
        with open(fname, 'r') as file: data = json.load(file)

        rels = set()
        ents = set()
        for d in tqdm(data): 
            r    = d['relation']
            toks = d['token']
            head = ' '.join(toks[d['subj_start']:d['subj_end']+1])
            tail = ' '.join(toks[d['obj_start']:d['obj_end']+1])

            rels.add(r)
            ents.update([head, tail])
        
        rels = list(rels)
        rels.remove(self.rn)
        rels = [self.rn] + sorted(rels)
        ents = sorted(list(ents))
        for i, r in enumerate(rels): self.rels[r] = i
        for i, e in enumerate(ents): self.ents[e] = i

        return   


class NYTPreprocessor(Preprocessor):
    def __init__(self, rn='NA'):
        super().__init__(rn)
        # don't know why NYT10 in some previous works excludes these relation types
        self.r_excluded = {
            '/business/company/industry', 
            '/business/company_shareholder/major_shareholder_of', 
            '/people/ethnicity/includes_groups', 
            '/people/ethnicity/people',
            '/sports/sports_team_location/teams'
        }
        return

    def load(self, ftrain, ftest, ftest_clean, valid_ratio=0.2, max_len=256, clean_nyt=False):
        print("load")
        with open(ftrain, 'r') as file: train = json.load(file)
        with open(ftest, 'r')  as file: test  = json.load(file)

        if clean_nyt:
            test_clean = []
            with open(ftest_clean, 'r') as file:
                lines = file.readlines()
                for l in lines:
                    test_clean.append(json.loads(l))


        print("build dict")
        self.update_dict(train)
        self.update_dict(test)

        print("build dataset")
        self.update_dataset(self.train, train, max_len=256)
        if clean_nyt:
            self.update_clean_dataset(self.test, test_clean, max_len=256)
        else:
            self.update_dataset(self.test, test, max_len=256)
        self.valid, self.train = self.train.split(ratios=[valid_ratio])
        return

    def update_dataset(self, dataset, data, max_len=256):
        for d in data:
            relation = d['relation']
            head = d['head']['word']
            tail = d['tail']['word']
            toks = d['sentence'].split()

            if len(toks) > max_len:
                continue

            if relation in self.r_excluded: continue

            locs = []
            for ent in [head, tail]:
                loc = []
                ent_toks = ent.split()
                for k in range(len(toks)-len(ent_toks)+1):
                    if toks[k] == ent_toks[0] and toks[k:k+len(ent_toks)] == ent_toks:
                        loc.append([k, k+len(ent_toks)-1])
                locs.append(loc)

            if len(locs[0]) * len(locs[1]) < 1:
                print("Error: zero mentions.")
                sys.exit(0)

            for head_loc in locs[0]:
                for tail_loc in locs[1]:
                    inst = {
                        'id': len(dataset),
                        'relation': {
                            'name': relation,
                            'id': self.rels[relation]
                        },
                        'head': {
                            'name': head,
                            'id': self.ents[head],
                            'start': head_loc[0],
                            'end': head_loc[1]
                        },
                        'tail': {
                            'name': tail,
                            'id': self.ents[tail],
                            'start': tail_loc[0],
                            'end': tail_loc[1]
                        },
                        'toks': toks
                    }
                    dataset.insert(inst)      
        return

    def update_dict(self, data):
        rels = set(self.rels.keys())
        ents = set(self.ents.keys())
        for d in data:
            relation = d['relation']
            head = d['head']['word']
            tail = d['tail']['word']

            if relation in self.r_excluded: continue

            rels.add(relation)
            ents.update([head, tail])

        rels = list(rels)
        rels.remove(self.rn)
        rels = [self.rn] + sorted(rels)
        ents = sorted(list(ents))
        for i, r in enumerate(rels): self.rels[r] = i
        for i, e in enumerate(ents): self.ents[e] = i
        return

    def update_clean_dataset(self, dataset, data, max_len=256):
        for d in data:
            for rel_mention in d['relationMentions']:
                # only keep clean data
                if rel_mention['is_noise'] == False:
                    relation = rel_mention['label']

                    # replace none type with the type we defined
                    if relation == 'None':
                        relation = self.rn

                    head = rel_mention['em1Text']
                    tail = rel_mention['em2Text']
                    toks = d['sentText'].split()

                    # Prevent KeyValue Error
                    if head not in self.ents or tail not in self.ents:
                        continue

                    if len(toks) > max_len:
                        continue

                    if relation in self.r_excluded:
                        continue

                    locs = []
                    for ent in [head, tail]:
                        loc = []
                        ent_toks = ent.split()
                        for k in range(len(toks)-len(ent_toks)+1):
                            if toks[k] == ent_toks[0] and toks[k:k+len(ent_toks)] == ent_toks:
                                loc.append([k, k+len(ent_toks)-1])
                        locs.append(loc)

                    if len(locs[0]) * len(locs[1]) < 1:
                        print("Error: zero mentions.")
                        sys.exit(0)

                    for head_loc in locs[0]:
                        for tail_loc in locs[1]:
                            inst = {
                                'id': len(dataset),
                                'relation': {
                                    'name': relation,
                                    'id': self.rels[relation]
                                },
                                'head': {
                                    'name': head,
                                    'id': self.ents[head],
                                    'start': head_loc[0],
                                    'end': head_loc[1]
                                },
                                'tail': {
                                    'name': tail,
                                    'id': self.ents[tail],
                                    'start': tail_loc[0],
                                    'end': tail_loc[1]
                                },
                                'toks': toks
                            }
                            dataset.insert(inst)      
        return


def preprocess_semeval():
    home = os.path.expanduser("~")
    f_path = "dataset/semeval2010/"
    f_train = "SemEval2010_task8_training/TRAIN_FILE.TXT"
    f_test  = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    folder = os.path.join(home, f_path)
    ftrain = os.path.join(folder, f_train)
    # fvalid = os.path.join(folder, fvalid)
    ftest  = os.path.join(folder, f_test)

    prep = SemEvalPreprocessor()
    prep.load(ftrain, ftest)
    # prep.load_from_processed(ftrain, fvalid, ftest)
    prep.dump_all(folder)

    dest_dir = os.path.join(folder, 'both')
    assure_folder_exist(dest_dir)
    for fn in range(0, 6):
        fn_ratio = fn * 0.05
        fp_ratio = fn * 0.05
        print('Synthesize noise ratio of:', fn_ratio + fp_ratio)
        prep.synth_noise(fn_ratio, fp_ratio)
        ftrain_synthed = os.path.join(dest_dir, f"train_{fn}.json")
        fvalid_synthed = os.path.join(dest_dir, f"valid_{fn}.json")
        prep.dump_dataset(ftrain=ftrain_synthed, fvalid=fvalid_synthed)
    return


def preprocess_tacred():
    home   = os.path.expanduser("~")    
    folder = "dataset/tacred"
    #ftrain = "data/json/train.json"
    #fvalid = "data/json/dev.json"
    #ftest  = "data/json/test.json"
    ftrain = "train_fn_5.json"
    fvalid = "valid_fn_5.json"
    ftest  = "test.json"

    folder = os.path.join(home, folder)
    ftrain = os.path.join(folder, ftrain)
    fvalid = os.path.join(folder, fvalid)
    ftest  = os.path.join(folder, ftest)

    prep   = TacredPreprocessor('no_relation')
    #prep.load(ftrain, fvalid, ftest)
    prep.load_from_processed(ftrain, fvalid, ftest)
    for fn in range(6, 10):
        fn_ratio = fn * 0.1
        prep.mask(fn_ratio)
        ftrain = os.path.join(folder, f"train_fn_{fn}.json")
        fvalid = os.path.join(folder, f"valid_fn_{fn}.json")
        prep.dump(ftrain=ftrain, fvalid=fvalid)
    return 

def preprocess_nyt():
    home        = os.path.expanduser("~")
    folder      = os.path.join(home, "dataset/nyt")
    ftrain      = os.path.join(folder, 'riedel/train.json')
    ftest       = os.path.join(folder, 'riedel/test.json')
    ftest_clean = '/share/home/cklee/dataset/nyt_clean/test.json'
    valid_ratio = 0.2

    prep = NYTPreprocessor()
    prep.load(ftrain, ftest, ftest_clean, valid_ratio=valid_ratio, max_len=256, clean_nyt=True)

    ftrain      = os.path.join(folder, 'train.json')
    fvalid      = os.path.join(folder, 'valid.json')
    ftest       = os.path.join(folder, 'test_clean.json')

    print('# of relations =', len(prep.rels))

    prep.dump(ftest=ftest)
    return


if __name__ == '__main__':
    preprocess_nyt()