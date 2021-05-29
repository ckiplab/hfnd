import os, sys
import numpy as np
import matplotlib.pyplot as plt

def precision_recall(targets, logits):
    # binerized targets
    binary_targets = np.zeros(logits.shape, dtype=int)
    binary_targets[np.arange(targets.shape[0]), targets] = 1

    # micro-average: flatten the positives
    binary_targets = binary_targets[:, 1:].flatten()
    binary_logits  = logits[:, 1:].flatten()

    # sort (from high to low probability)
    idxes = np.flip(np.argsort(binary_logits))
    binary_targets = binary_targets[idxes]
    binary_logits  = binary_logits[idxes]

    # compute precision and recall
    correct   = 0
    total     = np.sum(binary_targets)
    precision = np.zeros(binary_targets.shape[0])
    recall    = np.zeros(binary_targets.shape[0])
    for i, target in enumerate(binary_targets):
        correct += target 
        precision[i] = correct / (i+1)
        recall[i]    = correct / total

    return precision, recall


def per_inst_precision_recall(targets, logits):
    idxes   = np.flip(np.argsort(np.amax(logits[:, 1:], axis=1)))
    targets = targets[idxes]
    logits  = logits[idxes]
        
    # compute precision and recall
    correct   = 0
    total     = np.sum(targets != 0)
    precision = np.zeros(targets.shape[0])
    recall    = np.zeros(targets.shape[0])
    for i, target in enumerate(targets):
        if ((target != 0) and (target == np.argmax(logits[i]))):
            correct += 1 
        precision[i] = correct / (i+1)
        recall[i]    = correct / total

    return precision, recall


def micro_f1(targets, logits):
    correct   = 0
    selected  = 0
    relevent  = 0
    for target, logit in zip(targets, logits):
        r = np.argmax(logit)
        if target != 0 and target == r: correct += 1
        if target != 0: relevent += 1
        if r != 0: selected += 1

    if correct == 0 or selected == 0 or relevent == 0: 
        precision, recall, f1 = 0, 0, 0

    else:
        precision = correct / selected
        recall    = correct / relevent
        f1        = 2*precision*recall / (precision+recall)

    return f1, precision, recall


def plot_prc(dataset, arch, labels, colors, precisions, recalls, std, x_lim, hatch, marker, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    # fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(recalls[i], precisions[i], f'{colors[i]}{marker[i]}', label=labels[i], markersize=4, markevery=200)
        ax.fill_between(recalls[i], precisions[i] - std[i], precisions[i] + std[i], color=f'{colors[i]}', hatch=hatch[i], alpha=.2)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, x_lim + 0.05)

    plt.subplots_adjust(bottom=0.15)
    title = f"{arch.upper()} On {dataset.upper()}"
    plt.title(title, fontsize=18)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.legend(loc='lower left')
    plt.savefig(fname)
    return

def plot_nyt_logit():
    dataset = 'nyt'
    archs   = ['cnn']#, 'pcnn']
    # labels  = ['base', 'Co-teaching', 'Cleanlab', 'H-FND']
    labels = ['base', 'cleanlab', 'H-FND']
    colors  = ['r', 'b', 'k']

    selected_repeats = [
        [1, 2, 3],  # base
        [0, 1, 4],  # cleanlab
        [0, 2, 4]   # H-FND
    ]

    for arch in archs:
        home = os.path.expanduser("~")
        fplot = os.path.join(home, f"ckip_re/img/prc_{dataset}_{arch}_average.png")

        precisions = []
        recalls = []
        logit_sums = [None, None, None]
        for n_model in range(len(selected_repeats[0])):
            fpredicts = [
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}/test_{selected_repeats[0][n_model]}.npz"),
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}_cleanlab/test_{selected_repeats[1][n_model]}.npz"),
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}_rl/test_{selected_repeats[2][n_model]}.npz")
            ]

            # pass of all value in model
            for i, f_pred in enumerate(fpredicts):
                logit_sum = None
                with open(f_pred, 'rb') as file:
                    data = np.load(file)
                    targets = data['targets']
                    logits  = data['logits']
                    f1, precision, recall = micro_f1(targets, logits)
                    #print(f"f  = {fname}:")
                    #print(f"f1 = {f1}")
                    #print(f"p  = {precision}")
                    #print(f"r  = {recall}")
                    if logit_sums[i] is None:
                        logit_sums[i] = np.zeros_like(logits)
                    logit_sums[i] += logits

        for l in range(3):
            logits_avg = logit_sums[l] / 3
            precision, recall = per_inst_precision_recall(targets, logits_avg)

            precisions.append(precision)
            recalls.append(recall)
        
        plot_prc(dataset, arch, labels, colors, precisions, recalls, fplot)
    return

def plot_nyt_curve():
    dataset = 'nyt'
    archs   = ['cnn', 'pcnn']
    # labels  = ['base', 'Co-teaching', 'Cleanlab', 'H-FND']
    labels = ['base', 'cleanlab', 'H-FND']
    colors  = ['r', 'b', 'k']
    hatches = ['\\\\\\\\', '||||', '----']
    markers = ['-', 'o-', '^-']

    selected_repeats = {
        'cnn': [
            [1, 2, 3],  # base
            [0, 1, 4],  # cleanlab
            [0, 2, 4]   # H-FND
        ],
        'pcnn': [
            [1, 3, 4],  # base
            [1, 2, 3],  # cleanlab
            [2, 3, 4]   # H-FND
        ]
    }
    for arch in archs:
        home = os.path.expanduser("~")
        fplot = os.path.join(home, f"ckip_re/img/prc_{dataset}_{arch}_average.png")

        # precision_interps = [
        #     np.zeros(10000), # base
        #     np.zeros(10000), # cleanlab
        #     np.zeros(10000) # H-FND
        # ]
        precision_interps = [
            [], # base
            [], # cleanlab
            [] # H-FND
        ]

        rightmost_points = [
            [], # base
            [], # cleanlab
            []  # H-FND
        ]
        for n_model in range(len(selected_repeats[arch][0])):
            fpredicts = [
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}/test_{selected_repeats[arch][0][n_model]}.npz"),
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}_cleanlab/test_{selected_repeats[arch][1][n_model]}.npz"),
                os.path.join(home, f"ckip_re/predict/{dataset}_{arch}_rl/test_{selected_repeats[arch][2][n_model]}.npz")
            ]

            # pass of all value in model
            for n, f_pred in enumerate(fpredicts):
                logit_sum = None
                with open(f_pred, 'rb') as file:
                    data = np.load(file)
                    targets = data['targets']
                    logits  = data['logits']
                    precision, recall = per_inst_precision_recall(targets, logits)

                    steps = np.linspace(0, 1, 10000)
                    interpolate = [{'val': v, 'closest_dist': 1, 'original_index': None} for v in steps]
                    for i, r in enumerate(recall):
                        remain = int(r // 0.00005)
                        if remain % 2 == 0: # even
                            index = remain // 2
                        else:
                            index = (remain + 1) // 2
                            
                        if interpolate[index]['closest_dist'] > abs(r - remain * 0.00005):
                            interpolate[index]['closest_dist'] = abs(r - remain * 0.00005)
                            interpolate[index]['original_index'] = i

                    valid_points = []
                    rightmost_point = 0
                    for i, point in enumerate(interpolate):
                        if point['original_index'] is not None:
                            valid_points.append(point)
                            rightmost_point = i

                    sample_precisions = [precision[vp['original_index']] for vp in valid_points]
                    sample_recalls = [vp['val'] for vp in valid_points]

                    precision_interp = np.interp(steps, sample_recalls, sample_precisions)
                    precision_interp[rightmost_point:] = 0
                    precision_interps[n].append(precision_interp)
                    rightmost_points[n].append(valid_points[-1]['val'])
        
        
        precisions_final = []
        recalls_final = []
        stdev_final = []
        for n_model in range(len(selected_repeats[arch])):
            all_pres_interp = np.stack(precision_interps[n_model])
            precision_interp_avg = np.mean(all_pres_interp, axis=0)
            stdev = np.std(all_pres_interp, axis=0)
            plot_bound = max(rightmost_points[n_model])

            precisions_final.append(precision_interp_avg)
            recalls_final.append(np.linspace(0, 1, 10000))
            stdev_final.append(stdev)
        
        plot_prc(dataset, arch, labels, colors, precisions_final, recalls_final, stdev_final, plot_bound, hatches, markers, fplot)
    return


if __name__ == '__main__':
    plot_nyt_curve()
    # test()