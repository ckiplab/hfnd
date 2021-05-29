import numpy as np
import matplotlib.pyplot as plt
from visualization.expstat import ExpData, PolicyDistr

def plot_f1_errorbar(dataset, arch, rows, colors, fn, data, errors, fname, markers):
    fig, ax = plt.subplots(figsize=(6.4, 3.6)) # default (6.4, 4.8)
    # fig, ax = plt.subplots()
    for i in range(len(rows)):
        idxes = [k for k in range(len(data[i])) if data[i][k] is not None] 
        f = [fn[k] for k in idxes]
        m = [data[i][k] * 100  for k in idxes]
        e = [errors[i][k] * 100 for k in idxes]
        ax.errorbar(f, m, yerr=e, fmt=markers[i], color=colors[i], ecolor=colors[i], label=rows[i]) 
    
    plt.subplots_adjust(bottom=0.15)
    offset = (fn[1] - fn[0]) / 2
    plt.title(arch+' On '+dataset, fontsize=18)
    plt.xlabel('FN ratio (%)', fontsize=15)
    plt.ylabel('F1 score (%)', fontsize=15)
    plt.xlim(fn[0]-offset, fn[-1]+offset)
    plt.legend(loc='lower left')
    plt.savefig(fname)
    return

def plot_f1_errorbar_semeval():
    arch    = 'CNN'
    dataset = 'SemEval'
    colors  = ['r', 'tab:orange', 'g', 'b', 'c', 'k']    
    markers = ['o-', 'v-', '^-', '<-', '>-', 's-']
    rows    = ['base', 'SelATT', 'IRMIE', 'coteaching', 'cleanlab', 'H-FND']
 
    ed = ExpData()
    ed.load("../fnd_acl/result/log_semeval2010.txt", 'base')
    ed.load("../fnd_acl/result/log_semeval2010_att.txt", 'SelATT')
    ed.load("result/semeval2010_cnn_bootstrap_.txt", 'IRMIE')
    ed.load("result/semeval2010_pcnn_bootstrap.txt", 'IRMIE')
    ed.load("result/semeval2010_coteaching_fn.txt", 'coteaching')
    ed.load("result/semeval2010_cleanlab.txt", 'cleanlab')
    ed.load("../fnd_acl/result/log_semeval2010_pre_fnd.txt", 'H-FND')
        
    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_errorbar_semeval_cnn.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_errorbar_semeval_pcnn.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return

def plot_f1_errorbar_semeval_ablation():
    arch    = 'CNN'
    dataset = 'SemEval'
    colors  = ['k', 'tab:purple', 'tab:pink']
    markers = ['--', 'o-', '^-']
    rows    = ['H-FND', 'w/o revise', 'w/o pretrain'] 
    ed = ExpData()
    ed.load("../fnd_acl/result/log_semeval2010_pre_fnd.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_semeval_pre_fnd_nr.txt", 'w/o revise')
    ed.load("../fnd_acl/result/log_semeval2010_fnd.txt", 'w/o pretrain')
    

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_errorbar_semeval_cnn_ab.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_errorbar_semeval_pcnn_ab.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return

def plot_f1_errorbar_tacred():
    arch    = 'CNN'
    dataset = 'TACRED'
    colors  = ['r', 'tab:orange', 'g', 'b', 'c', 'k']
    markers = ['o-', 'v-', '^-', '<-', '>-', 's-']  
    rows    = ['base', 'SelATT', 'IRMIE', 'coteaching', 'cleanlab', 'H-FND'] 
    ed = ExpData()
    ed.load("../fnd_acl/result/log_tacred_cnn.txt"  , 'base')
    ed.load("../fnd_acl/result/log_tacred_att_0.txt", 'SelATT')
    ed.load("../fnd_acl/result/log_tacred_att_1.txt", 'SelATT')
    ed.load("../fnd_acl/result/log_tacred_att_2.txt", 'SelATT')
    ed.load("../fnd_acl/result/log_tacred_att_3.txt", 'SelATT')
    ed.load("../fnd_acl/result/log_tacred_att_4.txt", 'SelATT') 

    ed.load("result/tacred_bootstrap.txt", 'IRMIE')
    ed.load("result/tacred_coteaching_fn_single.txt", 'coteaching')
    ed.load("result/tacred_cleanlab.txt", 'cleanlab')
    ed.load("result/tacred_cleanlab_0.txt", 'cleanlab')
    ed.load("result/tacred_cleanlab_1.txt", 'cleanlab')
    ed.load("result/tacred_cleanlab_2.txt", 'cleanlab') 
    ed.load("result/tacred_cleanlab_3.txt", 'cleanlab') 
    ed.load("../fnd_acl/result/log_tacred_pre_pg.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_pre_pg_pcnn.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_pre_pg_rest.txt", 'H-FND')

    #ed.load("../result/tacred_cont_cnn.txt", 'base')
    ed.load("result/tacred_cont_coteaching.txt", 'coteaching')
    ed.load("result/tacred_cont_cleanlab.txt", 'cleanlab')
    ed.load("result/tacred_cont_cleanlab_0.txt", 'cleanlab')
    ed.load("result/tacred_cont_cleanlab_2.txt", 'cleanlab')
    ed.load("result/tacred_cont_cleanlab_3.txt", 'cleanlab')
    # ed.load("result/tacred_cont_cleanlab_iar.txt", 'cleanlab_iar')
    ed.load("../fnd_acl/result/log_tacred_cont_pre_fnd.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_cont_pre_fnd_1.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_cont_pre_fnd_2.txt", 'H-FND')

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_errorbar_tacred_cnn.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_errorbar_tacred_pcnn.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return

def plot_f1_errorbar_tacred_ablation():
    arch    = 'CNN'
    dataset = 'TACRED'
    colors  = ['k', 'tab:purple', 'tab:pink'] 
    markers = ['--', 'o-', '^-']   
    rows    = ['H-FND', 'w/o revise', 'w/o pretrain'] 
    ed = ExpData()
    ed.load("../fnd_acl/result/log_tacred_pre_pg.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_pre_pg_pcnn.txt", 'H-FND')
    ed.load("../fnd_acl/result/log_tacred_pre_pg_rest.txt", 'H-FND')

    ed.load("../fnd_acl/result/log_tacred_pre_fnd_nr.txt", 'w/o revise')
    ed.load("../fnd_acl/result/log_tacred_pg.txt", 'w/o pretrain')
    ed.load("../fnd_acl/result/log_tacred_pg_1.txt", 'w/o pretrain')
    ed.load("../fnd_acl/result/log_tacred_pg_2.txt", 'w/o pretrain')
    ed.load("../fnd_acl/result/log_tacred_pg_3.txt", 'w/o pretrain')
    

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_errorbar_tacred_cnn_ab.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_errorbar_tacred_pcnn_ab.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return


def plot_f1_valid_errorbar_semeval():
    arch    = 'CNN'
    dataset = 'SemEval'
    colors  = ['r', 'tab:orange', 'g', 'b', 'c', 'k']
    markers = ['o-', 'v-', '^-', '<-', '>-', 's-']   
    rows    = ['base', 'SelATT', 'IRMIE', 'coteaching', 'cleanlab', 'H-FND'] 
    ed = ExpData()
    ed.load("result_valid/semeval_cnn.txt", 'base', score='f1_valid')
    ed.load("result_valid/semeval_att.txt", 'SelATT', score='f1_valid')
    ed.load("result_valid/semeval_bootstrap.txt", 'IRMIE', score='f1_valid')
    ed.load("result_valid/semeval_coteaching.txt", 'coteaching', score='f1_valid')
    ed.load("result_valid/semeval_cleanlab.txt", 'cleanlab', score='f1_valid')
    ed.load("result_valid/semeval_pre_fnd.txt", 'H-FND', score='f1_valid')
        
    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_valid_errorbar_semeval_cnn.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_valid_errorbar_semeval_pcnn.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return

def plot_f1_valid_errorbar_semeval_ablation():
    arch    = 'CNN'
    dataset = 'SemEval'
    colors  = ['k', 'tab:purple', 'tab:pink']
    markers = ['--', 'o-', '^-']    
    rows    = ['H-FND', 'w/o revise', 'w/o pretrain'] 
    ed = ExpData()
    ed.load("result_valid/semeval_pre_fnd.txt", 'H-FND', score='f1_valid')
    ed.load("result_valid/semeval_pre_fnd_nr.txt", 'w/o revise', score='f1_valid')
    ed.load("result_valid/semeval_fnd.txt", 'w/o pretrain', score='f1_valid')
    

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_valid_errorbar_semeval_cnn_ab.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_valid_errorbar_semeval_pcnn_ab.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return

def plot_f1_valid_errorbar_tacred():
    arch    = 'CNN'
    dataset = 'TACRED'
    colors  = ['r', 'tab:orange', 'g', 'b', 'c', 'y', 'k']
    markers = ['o-', 'v-', '^-', '<-', '>-', 's-']     
    rows    = ['base', 'SelATT', 'IRMIE', 'coteaching', 'cleanlab', 'H-FND'] 
    ed = ExpData()
    ed.load("result_valid/tacred_cnn.txt", 'base', score='f1_valid')
    ed.load("result_valid/tacred_att.txt", 'SelATT', score='f1_valid')
    ed.load("result_valid/tacred_bootstrap.txt", 'IRMIE', score='f1_valid')
    ed.load("result_valid/tacred_coteaching.txt", 'coteaching', score='f1_valid')
    ed.load("result_valid/tacred_cleanlab.txt", 'cleanlab', score='f1_valid')
    ed.load("result_valid/tacred_pre_fnd.txt", 'H-FND', score='f1_valid')

    #ed.load("../result/tacred_cont_cnn.txt", 'base')
    ed.load("result_valid/tacred_cont_coteaching.txt", 'coteaching', score='f1_valid')
    ed.load("result_valid/tacred_cont_cleanlab.txt", 'cleanlab', score='f1_valid')
    ed.load("result_valid/tacred_cont_pre_fnd.txt", 'H-FND', score='f1_valid')

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_valid_errorbar_tacred_cnn.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_valid_errorbar_tacred_pcnn.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return


def plot_f1_valid_errorbar_tacred_ablation():
    arch    = 'CNN'
    dataset = 'TACRED'
    colors  = ['k', 'tab:purple', 'tab:pink']
    markers = ['--', 'o-', '^-']  
    rows    = ['H-FND', 'w/o revise', 'w/o pretrain'] 
    ed = ExpData()
    ed.load("result_valid/tacred_pre_fnd.txt", 'H-FND', score='f1_valid')
    ed.load("result_valid/tacred_pre_fnd_nr.txt", 'w/o revise', score='f1_valid')
    ed.load("result_valid/tacred_fnd.txt", 'w/o pretrain', score='f1_valid')

    fn, data, errors = ed.get_stat([f"{r}_cnn" for r in rows])
    fname = 'img/f1_valid_errorbar_tacred_cnn_ab.png'
    plot_f1_errorbar(dataset, 'CNN', rows, colors, fn, data, errors, fname, markers)
    
    fn, data, errors = ed.get_stat([f"{r}_pcnn" for r in rows])
    fname = 'img/f1_valid_errorbar_tacred_pcnn_ab.png'
    plot_f1_errorbar(dataset, 'PCNN', rows, colors, fn, data, errors, fname, markers)
    return


def plot_bar(dataset, arch, colors, rows, columns, data, error, fname, hatches):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))#figsize=(9, 5))
    x_unit  = columns[1] - columns[0]
    width   = 0.2 * x_unit
    x_sep   = 0.1 * x_unit

    n_rows  = len(rows)
    index   = np.arange(len(columns)) * x_unit
    y_offset = np.zeros(len(columns)) * x_unit

    for row in range(n_rows):
        if row <  n_rows//2: x = index - width/2 - x_sep/2
        else:                x = index + width/2 + x_sep/2
        if row == n_rows//2: y_offset = np.zeros(len(columns))
        y = [d * 100 for d in data[row]]
        #ax.bar(x, y, width, yerr=e, bottom=y_offset, color=colors[row], label=rows[row])
        ax.bar(x, y, width, bottom=y_offset, color=colors[row], label=rows[row], alpha=.8, hatch=hatches[row])
        y_offset = y_offset + y

    plt.subplots_adjust(bottom=0.15)
    plt.rcParams['hatch.linewidth'] = 0.5
    plt.xlabel('FN ratio (%)', fontsize=15)
    plt.ylabel('Percentage of actions (%)', fontsize=15)
    plt.xlim(columns[0] -1.0 * x_unit, columns[-1] + 0.5 * x_unit)
    plt.xticks(columns)
    #plt.yticks([])
    plt.title(f"Policy Distribution, {arch} on {dataset}", fontsize=18)
    ax.legend()
    #plt.show()
    plt.savefig(fname)
    return

def plot_policy_distr_semeval():
    arch    = 'CNN'
    dataset = 'SemEval'
    colors  = np.concatenate((plt.cm.Blues(np.linspace(0.25, 0.75, 3)), 
                              plt.cm.Reds(np.linspace(0.25, 0.75, 3))), axis=0)
    rows    = ['TN/Keep', 'TN/Discard', 'TN/Revise', 'FN/Keep', 'FN/Discard', 'FN/Revise']
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx']

    pd = PolicyDistr()
    pd.load("result/semeval_pd.json")
    fn_list, distr_table, error_table = pd.get_stat('cnn')
    columns = [int(f)*10 for f in fn_list]
    data    = np.reshape(distr_table, (len(columns), -1)).T
    error   = np.reshape(error_table, (len(columns), -1)).T
    fname = 'img/policy_distr_semeval_cnn.png'
    plot_bar(dataset, 'CNN', colors, rows, columns, data, error, fname, hatches)

    fn_list, distr_table, error_table = pd.get_stat('pcnn')
    columns = [int(f)*10 for f in fn_list]
    data    = np.reshape(distr_table, (len(columns), -1)).T
    error   = np.reshape(error_table, (len(columns), -1)).T
    fname = 'img/policy_distr_semeval_pcnn.png'
    plot_bar(dataset, 'PCNN', colors, rows, columns, data, error, fname, hatches)
    return

def plot_policy_distr_tacred():
    arch    = 'CNN'
    dataset = 'TACRED'
    colors  = np.concatenate((plt.cm.Blues(np.linspace(0.25, 0.75, 3)), 
                              plt.cm.Reds(np.linspace(0.25, 0.75, 3))), axis=0)
    rows    = ['TN/Keep', 'TN/Discard', 'TN/Revise', 'FN/Keep', 'FN/Discard', 'FN/Revise']
    hatches = ['///', '\\\\\\', '|||', '---', '+++', 'xxx']

    pd = PolicyDistr()
    pd.load("result/tacred_pd.json")
    fn_list, distr_table, error_table = pd.get_stat('cnn')
    columns = [int(f)*10 for f in fn_list]
    data    = np.reshape(distr_table, (len(columns), -1)).T
    error   = np.reshape(error_table, (len(columns), -1)).T
    fname = 'img/policy_distr_tacred_cnn.png'
    plot_bar(dataset, 'CNN', colors, rows, columns, data, error, fname, hatches)

    fn_list, distr_table, error_table = pd.get_stat('pcnn')
    columns = [int(f)*10 for f in fn_list]
    data    = np.reshape(distr_table, (len(columns), -1)).T
    error   = np.reshape(error_table, (len(columns), -1)).T
    fname = 'img/policy_distr_tacred_pcnn.png'
    plot_bar(dataset, 'PCNN', colors, rows, columns, data, error, fname, hatches)
    return


if __name__ == '__main__':
    # plot_f1_errorbar_semeval()
    # plot_f1_errorbar_tacred()
    # plot_f1_errorbar_semeval_ablation()
    # plot_f1_errorbar_tacred_ablation()
    plot_f1_valid_errorbar_semeval()
    plot_f1_valid_errorbar_tacred()
    plot_f1_valid_errorbar_semeval_ablation()
    plot_f1_valid_errorbar_tacred_ablation()
    # plot_policy_distr_semeval()
    # plot_policy_distr_tacred()
 