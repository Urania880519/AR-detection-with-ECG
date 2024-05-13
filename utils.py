import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, precision_recall_curve, f1_score, accuracy_score, roc_auc_score, average_precision_score
import wfdb
from sklearn.utils import resample


def split_data(seed=42):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]


def prepare_input(ecg_file: str):
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-15000:, :]
    result = np.zeros((15000, nleads)) # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.transpose()

import csv
def prepare_input2(ecg_file: str):
    ecg_data = []
    with open(ecg_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        head = next(reader)
        for ld in reader:
            ecg_data.append([float(x) for x in ld[1:]])
    # ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
    ecg_data_15 = np.zeros((15, 1238))
    
    ecg_data_15[0] = ecg_data[0][0:1238]
    ecg_data_15[1] = ecg_data[-1][0:1238]
    ecg_data_15[2] = ecg_data[2][0:1238]
    
    ecg_data_15[3] = ecg_data[-1][1250:1250+1238]
    ecg_data_15[4] = ecg_data[3][0:1238]
    ecg_data_15[5] = ecg_data[4][0:1238]
    ecg_data_15[6] = ecg_data[5][0:1238]


    ecg_data_15[7] = ecg_data[-1][2500:2500+1238]
    ecg_data_15[8] = ecg_data[6][0:1238]
    ecg_data_15[9] = ecg_data[7][0:1238]
    ecg_data_15[10] = ecg_data[8][0:1238]

    try:
            ecg_data_15[11] = ecg_data[-1][3750:3750+1238]
    except ValueError:
        ecg_data_15[11][:len(ecg_data[-1][3750:])] = ecg_data[-1][3750:]
    ecg_data_15[12] = ecg_data[9][0:1238]
    ecg_data_15[13] = ecg_data[10][0:1238]
    ecg_data_15[14] = ecg_data[11][0:1238]

    # ecg_data = np.array(ecg_data[:-1])
    ecg_data = np.transpose(ecg_data_15)
    # print(ecg_data.shape)
    # ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
    nsteps, _ = ecg_data.shape
    # ecg_data = ecg_data[-5000:, self.use_leads]
    result = np.zeros((1238, 15)) # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data

    return result.transpose()


def cal_scores(y_true, y_pred, y_score):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, auc, acc


def find_optimal_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return thresholds[np.argmax(f1s)]


def cal_f1(y_true, y_score, find_optimal):
    if find_optimal:
        thresholds = np.linspace(0, 1, 100)    
    else:
        thresholds = [0.5]
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return np.max(f1s)


def cal_f1s(y_trues, y_scores, find_optimal=True):
    f1s = []
    for i in range(y_trues.shape[1]):
        f1 = cal_f1(y_trues[:, i], y_scores[:, i], find_optimal)
        f1s.append(f1)
    return np.array(f1s)


def cal_aucs(y_trues, y_scores):
    return roc_auc_score(y_trues, y_scores, average=None)

def cal_auprc(y_trues, y_scores):
    return average_precision_score(y_trues, y_scores, average= None)
    
def cal_cis(y_trues, y_scores, targets, _auc= True, _auprc= False):
    all_cis=[]
    for i in range(len(targets)):
        _scores= []
        y_true= y_trues[:, i]
        y_score= y_scores[:, i]
        n_bootstraps= 1000
        for _ in range(n_bootstraps):
            boot_indices= resample(range(len(y_true)), n_samples=len(y_true))
            boot_true = y_true[boot_indices]
            boot_probs = y_score[boot_indices]
            if _auc:
                fpr, tpr, _ = roc_curve(boot_true, boot_probs)
                _scores.append(auc(fpr, tpr))
            elif _auprc:
                precision, recall, _ = precision_recall_curve(boot_true, boot_probs)  
                _scores.append(auc(recall, precision))
            else:
                print('wrong kwargs')
        confidence_level= 0.95
        lower_bound = np.percentile(_scores, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(_scores, (1 + confidence_level) / 2 * 100)
        all_cis.append((lower_bound, upper_bound))
    return all_cis



    
