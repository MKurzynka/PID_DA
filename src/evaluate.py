import torch 
from tqdm import tqdm
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
        false_pos = sum(incorrect[true_pdg_code != i])
        false_neg = sum(incorrect[true_pdg_code == i])
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
