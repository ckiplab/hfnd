import os, sys
import json
import numpy as np

class ExpData:
    def __init__(self):
        self.data = {}
        return

    def load(self, fname, model='org', score='f1_test'):
        file = open(fname, 'r')
        for line in file:
            d = {}
            line = line.strip()
            fields = line.split(',')
            for field in fields:
                key, value = field.split('=')
                d[key.strip()] = value.strip()

            arch = f"{model}_{d['arch']}"
            fn   = int(d['fn'])
            f1   = float(d[score])
            if arch not in self.data: self.data[arch] = {}
            if fn   not in self.data[arch]: self.data[arch][fn] = []
            self.data[arch][fn].append(f1)
        file.close()
        return

    def get_stat(self, archs=[]):
        fn_list    = set()
        for arch in archs: 
            if arch in self.data: 
                fn_list |= set(self.data[arch].keys())
        fn_list    = sorted(list(fn_list))
        mean_table = []
        std_table  = []

        for arch in archs:
            means = []
            stds  = []
            for fn in fn_list:
                if arch not in self.data or fn not in self.data[arch]: 
                    means.append(None)
                    stds.append(None)
                    continue

                data = self.data[arch][fn]
                l    = min(5, len(data))
                data = np.array(data[:l])
                mean = 0 if l == 0 else np.mean(data)
                std  = 0 if l == 0 else np.std(data)
                means.append(mean)
                stds.append(std)
                
            mean_table.append(means)
            std_table.append(stds)

        fn_list = [fn * 10 for fn in fn_list]
        return fn_list, mean_table, std_table

    def print_stat(self):
        for arch in self.data:
            for fn in sorted(list(self.data[arch].keys())):
                data = self.data[arch][fn]
                l    = min(5, len(data))
                data = np.array(data[:l])
                mean = 0 if l == 0 else np.mean(data)
                std  = 0 if l == 0 else np.std(data)
                pad  = "" if l == 5 else " " + " ".join(["."]*(5-l))
                print(f"{arch} {fn} " + " ".join([f"{d:.4f}" for d in data]) + pad + f" {mean:.4f} {std:.4f}")
        return

class PolicyDistr:
    def __init__(self):
        self.data = {}
        return

    def load(self, fname):
        with open(fname, 'r') as file: data = json.load(file)

        for d in data:
            arch  = d['arch']
            fn    = d['fn']
            distr = d['distr']
            if arch not in self.data: self.data[arch] = {}
            if fn not in self.data[arch]: self.data[arch][fn] = []
            self.data[arch][fn].append(distr)
        return

    def get_stat(self, arch):
        data = self.data[arch]
        fn_list = sorted(list(data.keys()))
        distr_table = []
        error_table = []
        for fn in fn_list:
            m = np.mean(data[fn], axis=0)
            e = np.std(data[fn], axis=0)
            #m = np.median(data[fn], axis=0)
            #s = np.sum(m, axis=-1)
            #for i in range(s.shape[0]):
            #    if s[i] != 0.0: m[i] /= s[i]
            distr_table.append(m)
            error_table.append(e)

        return fn_list, distr_table, error_table


def print_table():
    pd = PolicyDistr()
    #pd.load("../semeval_pd.json")
    pd.load("../tacred_pd.json")
    fn_list, distr_table = pd.get_stat('cnn')
    print(fn_list)
    print(np.array(distr_table))
    return


def latex_policy_table(pd, arch):
    rows = ['TN/Keep', 'TN/Discard', 'TN/Revise', 'FN/Keep', 'FN/Discard', 'FN/Revise']
    fn_list, distr_table, error_table = pd.get_stat(arch)

    columns = [int(f)*10 for f in fn_list]
    data    = np.reshape(distr_table, (len(columns), -1)).T
    error   = np.reshape(error_table, (len(columns), -1)).T

    for i, row in enumerate(rows):
        print(" & ".join([row] + [f"{d*100:.2f} $\\pm$ {e*100:.2f}" for d, e in zip(data[i], error[i])])
                + " \\\\")
    return

def latex_policy_table_semeval():
    pd = PolicyDistr()
    pd.load("../semeval_pd.json")
    print("semeval, cnn:")
    latex_policy_table(pd, 'cnn')
    print("semeval, pcnn:")
    latex_policy_table(pd, 'pcnn')
    return

def latex_policy_table_tacred():
    pd = PolicyDistr()
    pd.load("../tacred_pd.json")
    print("tacred, cnn:")
    latex_policy_table(pd, 'cnn')
    print("tacred, pcnn:")
    latex_policy_table(pd, 'pcnn')
    return

if __name__ == '__main__':
    latex_policy_table_semeval()
    latex_policy_table_tacred()