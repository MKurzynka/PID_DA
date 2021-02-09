import torch 
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate
from matplotlib.pyplot import figure, axes

def propagte_data_through_network(data_loader, network, device):
    out = torch.cuda.FloatTensor()
    ture_l = torch.cuda.FloatTensor()
    for x, y, _ in tqdm(data_loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = network(x)
        particles = y_pred.detach()
        particles[particles>0.5] = 1
        particles[particles<=0.5] = 0
        out = torch.cat([out, particles], 0)
        ture_l = torch.cat([ture_l, y.view(-1)], 0)

    return out, ture_l

def evaluate_model_ova(pdg_code, true_pdg_code):
    copied_pdg_tmp = pdg_code.copy()
    copied_true_pdg_tmp = true_pdg_code.copy()
    correct = copied_pdg_tmp == copied_true_pdg_tmp
    incorrect = copied_pdg_tmp != copied_true_pdg_tmp
    true_pos = sum(correct[true_pdg_code == 1])
    false_pos = sum(incorrect[true_pdg_code != 1])
    false_neg = sum(incorrect[true_pdg_code == 1])
    true_neg = sum(correct[true_pdg_code != 1])
    acc = (true_pos + true_neg)/len(pdg_code)
    prec = true_pos/(true_pos + false_pos)
    sens = true_pos/(true_pos + false_neg)
    spec = true_neg/(false_pos + true_neg)
    f1 = 2*(true_pos/(true_pos + false_pos)*true_pos/(true_pos + false_neg))/(true_pos/(true_pos + false_pos) + true_pos/(true_pos + false_neg))
    stats = [acc, prec, sens, spec, f1]    
    print("True positive #: " + str(true_pos))
    print("False positive #: " + str(false_pos))
    print("False negative #: " + str(false_neg))
    print("True negative #: " + str(true_neg))
    print("Accuracy: " + str(acc))
    print("Precision: " + str(prec))
    print("Sensitivity: " + str(sens))
    print("Specificity: " + str(spec))
    print("F1: " + str(f1))
    return stats
    

def evaluate_model(pdg_code, true_pdg_code):
    stats = {}
    for i in range(4):
        print("########### " + str(i) + " ###########")
        copied_pdg_tmp = pdg_code.copy()
        copied_true_pdg_tmp = true_pdg_code.copy()
        copied_pdg_tmp[pdg_code != i] = -1
        copied_true_pdg_tmp[true_pdg_code != i] = -1
        correct = copied_pdg_tmp == copied_true_pdg_tmp
        incorrect = copied_pdg_tmp != copied_true_pdg_tmp
        true_pos = sum(correct[true_pdg_code == i])
        false_pos = sum(incorrect[true_pdg_code == i])
        false_neg = sum(incorrect[true_pdg_code != i])
        true_neg = sum(correct[true_pdg_code != i])
        acc = (true_pos + true_neg)/len(pdg_code)
        prec = true_pos/(true_pos + false_pos)
        sens = true_pos/(true_pos + false_neg)
        spec = true_neg/(false_pos + true_neg)
        f1 = 2*(true_pos/(true_pos + false_pos)*true_pos/(true_pos + false_neg))/(true_pos/(true_pos + false_pos) + true_pos/(true_pos + false_neg))
        stats[i] = [acc, prec, sens, spec, f1]    
        print("True positive #: " + str(true_pos))
        print("False positive #: " + str(false_pos))
        print("False negative #: " + str(false_neg))
        print("True negative #: " + str(true_neg))
        print("Accuracy: " + str(acc))
        print("Precision: " + str(prec))
        print("Sensitivity: " + str(sens))
        print("Specificity: " + str(spec))
        print("F1: " + str(f1))
    
    return stats

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))

def one_gaussians(x, h1, c1, w1):
    return (gaussian(x, h1, c1, w1))

def estimate_classification_quality(data_frame, pdg_code):
    positive_classified = data_frame.query('pdg_code == 1')
    negative_classified = data_frame.query('pdg_code == 0')
    n_ranges = 4
    p_ranges = [0.48272076, 0.58704057, 0.69136038, 0.79568019, 0.9]
    
    df_array_pos = []
    df_array_neg = []
    counts_0 = []
    counts_1 = []
    bins = []
    p = []
    p_res = []
    
    true_positives = []
    false_negative = []
    not_included = []
    false_positives = []
    spec = []
    
    optim = np.array([[9.28490710e-02, 4.92932568e+01, 4.5009191e+00, 2.006539930e-03,
         8.4027208e+01, 5.6641913e+00, 1.25670632e-03, 8.45701248e+01,
         4.81806393e+00, 2.119629896e-04, 1.89537870e+02, 1.29678890e+01],
        [9.65255222e-02, 5.03579685e+01, 3.9564949e+00, 3.31816877e-03,
         7.16847763e+01, 5.60187743e+00, 1.27833193e-03, 8.41476779e+01,
         5.32993537e+00, 1.21022554e-03, 1.59861195e+02, 1.61194061e+01],
        [8.53949927e-02, 5.0438144e+01, 4.29894069e+00, 4.76371132e-03,
         6.36489799e+01, 4.96055621e+00, 1.07104088e-03, 8.38109634e+01,
         5.10032287e+00, 3.1240542e-03, 1.26808938e+02, 1.00731521e+01],
        [8.12346697e-02, 5.11918945e+01, 4.52243773e+00, 7.08224843e-03,
         5.95458302e+01, 4.56349196e+00, 1.01306200e-03, 8.40796827e+01,
         5.35597779e+00, 4.25516412e-03, 1.03328981e+02, 9.37278245e+00]])
    
    for i in range(n_ranges):
        spec.append(optim[i][3*pdg_code : 3*pdg_code + 3])
        df_array_pos.append(positive_classified[
            positive_classified['P'].between(p_ranges[i], p_ranges[i + 1], inclusive=False)])
        df_array_neg.append(negative_classified[
            negative_classified['P'].between(p_ranges[i], p_ranges[i + 1], inclusive=False)])
        
        fig = plt.figure(figsize=(15, 8), dpi=80)
        ax = axes()
        
        bins_tmp = np.linspace(25, 250, 100)
        counts_tmp, bins_tmp, _ = plt.hist([df_array_neg[i].tpc_signal, df_array_pos[i].tpc_signal], bins=bins_tmp, 
                                           alpha=0.9, edgecolor='black',  density=True, stacked=True)
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.7)
        plt.xlim((0, 250))
        plt.ylim((9e-5, 1))
        plt.xlabel('tpc_signal')
        plt.yscale('log')
        ax.set_facecolor('w')
        plt.title('tpc_signal distribution, P in range (' + str(round(p_ranges[i], 2)) + ', ' + str(round(p_ranges[i + 1], 2)) +']')
        
        del fig, ax
        plt.pause(0.1)
        counts_0.append(counts_tmp[0])
        counts_1.append(counts_tmp[1])

        new_bin = []
        for j in range(len(bins_tmp) - 1):
            new_bin.append(bins_tmp[j] + (bins_tmp[j + 1] - bins_tmp[j])/2)

        bins.append(new_bin)

        p_tmp = integrate.quad(lambda x: one_gaussians(x, *optim[i][3*pdg_code:3*pdg_code+3]), 0, 300)        

        p.append(p_tmp[0])
        p_res.append(len(df_array_pos[i])/(len(df_array_neg[i]) + len(df_array_pos[i])))
    

        current_data_frame = df_array_pos[i]
        current_data_frame_not_2212 = df_array_neg[i]
        true_positives.append(current_data_frame[
            current_data_frame['tpc_signal'].between(spec[i][1]-3*spec[i][2], spec[i][1]+3*spec[i][2],
                                                     inclusive=True)])
        false_negative.append(current_data_frame_not_2212[
            current_data_frame_not_2212['tpc_signal'].between(
                spec[i][1]-3*spec[i][2], spec[i][1]+3*spec[i][2], inclusive=True)])

        not_included.append(current_data_frame_not_2212[
            ~current_data_frame_not_2212['tpc_signal'].between(
                spec[i][1]-3*spec[i][2], spec[i][1]+3*spec[i][2], inclusive=True)])

        false_positives.append(current_data_frame[
            ~current_data_frame['tpc_signal'].between(
                spec[i][1]-3*spec[i][2], spec[i][1]+3*spec[i][2], inclusive=True)])
        print("range: " + str(i))
        print("Gauss center: " + str(spec[i][1]) + " sigma: " + str(optim[i][pdg_code + 3]))
        print("True positive: " + str(len(true_positives[i])))
        print("False positives: " + str(len(false_positives[i])))
        print("False negatives: " + str(len(false_negative[i])))
        print("True negatives: " + str(len(not_included[i])))
        print("Gauss probability : " + str(p[i]) + " Model probability: " + str(p_res[i]))
        print("Efficiency : " + str(p_res[i]/p[i]))
        print("Accuracy : " + str((len(true_positives[i])+ len(not_included[i]))/
                                  (len(df_array_pos[i]) + len(df_array_neg[i]))))

        fig = plt.figure(figsize=(15, 8), dpi=80)
        ax = axes()
        plt.hist(bins[i], bins[i], weights=(counts_1[i] - counts_0[i]), 
                                           alpha=0.9, edgecolor='black')
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.7)
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.7)
        plt.xlim((0, 250))
        plt.ylim((9e-5, 1))
        plt.xlabel('tpc_signal')
       
        plt.title('tpc_signal distribution, P in range (' + str(round(p_ranges[i], 2)) + ', ' + str(round(p_ranges[i + 1], 2)) +']')
        plt.plot(bins[i], one_gaussians(bins[i], *optim[i][3*pdg_code:3*pdg_code+3]), linewidth=2)
        plt.yscale('log')
        ax.set_facecolor('w')
        plt.pause(0.1)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")