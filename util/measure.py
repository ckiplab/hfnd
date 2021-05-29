#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
import numpy as np
import json
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 

class OldMicroF1:
    def __init__(self):
        self.correct  = 0
        self.selected = 0
        self.relevent = 0
    
    def get(self, full=False):
        if self.correct == 0 or self.selected == 0 or self.relevent == 0: 
            precision, recall, f1 = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        else:
            precision = self.correct/self.selected
            recall    = self.correct/self.relevent
            f1        = 2*precision*recall/(precision+recall)
        
        if full: return f1, precision, recall
        return f1
    
    def add(self, ro, ra, wgt=1.0):
        self.correct  += (((ro!=0) * (ro==ra)).float() * wgt).sum()
        self.selected += ((ro!=0).float() * wgt).sum()
        self.relevent += ((ra!=0).float() * wgt).sum()
        return


class MicroF1:
    """ Calculates the Micro F1-score.
        Currently implements 2 levels, instance-level F1-score (normal F1 implementation) or bag-level for better resilience to noise.
    
        methods:
            add(): Insert instance into respective bags, indexed by (head, tail)
            get(): Get the F1 score over all the instances that have been added
    """

    def __init__(self):
        self.bags = {}
        return

    def get(self, full=False, bagging=False):
        """ Get the F1 score of all the entries that have been added to the class so far.

            Parameter:
                full (boolean): If True, return f1 plus precision and recall if, otherwise return only f1-score.
                bagging (boolean): If True, return the bag-level f1, otherwise return instance-level f1.
        """
        correct  = torch.tensor(0.0)
        selected = torch.tensor(0.0)
        relevent = torch.tensor(0.0)

        if bagging is True:
            for (relation_ans, head, tail), bag in self.bags.items(): # iterate thru each bag. Each bag denotes a distinct (relation, head, tail)
                # Concat all the instance in the bag into one large tensor
                bag = torch.cat(bag, dim=0)

                # Get the largest value of each column across all instances in the bag
                out = torch.max(bag, dim=0)[0]

                # Get the index of the largest among "out"
                # The result ("relation_out") is the relation that has the highest confidence in the bag
                confidence, relation_out = torch.max(out, dim=0)

                if confidence == 0: continue
                if relation_out != 0 and relation_out == relation_ans: # correct prediction
                    correct += 1 # True positive
                if relation_out != 0:
                    selected += 1 # False positive
                if relation_ans != 0:
                    relevent += 1 # False negative
        else:
            for (relation_ans, head, tail), bag in self.bags.items():
                instance_logits = torch.cat(bag, dim=0)
                relation_out = torch.argmax(instance_logits, dim=1).cpu()
                relation_ans = relation_ans.expand(relation_out.size()).cpu()
                wgt = torch.sum(instance_logits, dim=1).cpu()
                
                correct  += (((relation_out != 0) & (relation_out == relation_ans)).float() * wgt).sum()
                selected += ((relation_out != 0).float() * wgt).sum()
                relevent += ((relation_ans != 0).float() * wgt).sum()

        if correct == 0 or selected == 0 or relevent == 0: 
            precision, recall, f1 = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        else:
            precision = correct / selected
            recall    = correct / relevent
            f1        = 2 * precision * recall / (precision + recall)

        
        if full:
            return f1, precision, recall
        else:
            return f1

    def add(self, logits, r, h, t, wgt=1.0):
        # the discarded instances have logits zeroed
        logits = logits * wgt

        for i in range(logits.size(0)):
            k = (r[i], h[i], t[i])
            if k not in self.bags:
                self.bags[k] = []
            self.bags[k].append(logits[i:i+1])
        return

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0
        return

    def add(self, ro, ra):
        self.correct += torch.sum((ro==ra).float())
        self.total   += len(ra)
        return

    def get(self):
        return self.correct / self.total 
