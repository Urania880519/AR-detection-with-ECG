import os
import pickle
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import numpy as np
from resnet import resnet34
from utils import cal_f1s, cal_aucs,cal_cis, split_data, find_optimal_threshold, cal_scores, cal_auprc
from dataset_mod import ECG_AR_Retrain
from test_retrain import scaling, shift, transform, ECG_AR_JP, test_scores, get_thresholds, apply_thresholds

data_dir= "/raid/data/tewei/ExVal_JP/signal_csv/"
targets = ['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEF(bp)<40%', 'LVEDD>65', 'LVESDi>20', 'LVESDi>25', 'LVESDi>30', 'TRPG>30', 'TRPG>40', 'LVMi>167', 'EDVi>93', 'ESVi>37', 'ESVi>45', 'BAV', 'NYHA>=3', 'CCI>=3', 'AR_VC>=6', 'AR_VC>=8']
criterion = nn.BCEWithLogitsLoss()
device= 'cuda'
leads= 'all'
folds = range(2, 7)
JP_label= "/data/tewei/ExVal_JP/JP_transfer_label.csv"
TW_label= "/data/tewei/ExVal_JP/AR_labels_cv_5_fold_test_79_verified.csv"
all_val_label= "/data/tewei/ExVal_JP/AllVal_transfer_label.csv"

csvfile = open(f'./transfer_result/Retrain_test_result_transfer.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['Fold', 'Target']+targets)

csvfile2 = open(f'./transfer_result/Retrain_thresholds_transfer.csv', 'w', newline='')
writer2 = csv.writer(csvfile2)
writer2.writerow(['Fold']+targets)

nets = []
y_trues_jp = []
y_scores_jp = []
y_trues_tw = []
y_scores_tw = []
thresholds_list = []
for s, fd in enumerate(folds):
    val_folds= np.array([fd])
    test_folds= np.array([1])
    
    JP_test_ds= ECG_AR_Retrain('test', data_dir, JP_label, test_folds, leads, targets)
    JP_test_loader= DataLoader(JP_test_ds, batch_size= 16, shuffle= False, num_workers= 4, pin_memory= True)
    TW_test_ds= ECG_AR_Retrain('test', data_dir, TW_label, test_folds, leads, targets)
    TW_test_loader= DataLoader(TW_test_ds, batch_size= 16, shuffle= False, num_workers= 4, pin_memory= True)
    
    val_dataset= ECG_AR_Retrain('validation', data_dir, all_val_label, val_folds, leads, targets)
    val_loader= DataLoader(val_dataset, batch_size= 16, shuffle= False, num_workers= 4, pin_memory= True)
    net = resnet34(input_channels=15, num_classes=len(targets)).to(device)
    state_dict= torch.load(f'./models/Finetuned_{s}.pth')
    net.load_state_dict(state_dict)
    net.eval()
    nets.append(net)

    y_trues, y_scores = test_scores(JP_test_loader, net, device)
    y_scores_jp.append(y_scores.tolist())
    y_trues_jp.append(y_trues.tolist())
    aucs = cal_aucs(y_trues, y_scores)
    auprcs= cal_auprc(y_trues, y_scores)
    print(aucs)
    writer.writerow([s, 'AUC_JP']+aucs.tolist())
    writer.writerow([s, 'AUPRC_JP']+auprcs.tolist())
    csvfile.flush()

    y_t, y_s = test_scores(TW_test_loader, net, device)
    y_scores_tw.append(y_s.tolist())
    y_trues_tw.append(y_t.tolist())
    aucs = cal_aucs(y_t, y_s)
    auprcs= cal_auprc(y_t, y_s)
    print(aucs)
    writer.writerow([s, 'AUC_TW']+aucs.tolist())
    writer.writerow([s, 'AUPRC_TW']+auprcs.tolist())
    csvfile.flush()

    thresholds = get_thresholds(val_loader, net, device)
    writer2.writerow([s]+thresholds)
    thresholds_list.append(thresholds)
    print('Thresholds:', thresholds)
    print('Results on mixed data:')
    apply_thresholds(JP_test_loader, net, device, thresholds)
    apply_thresholds(TW_test_loader, net, device, thresholds)
    torch.cuda.empty_cache()
    gc.collect()
    
y_scores_jp= np.array(y_scores_jp)
y_scores_tw= np.array(y_scores_tw)
y_scores_avg_jp= np.mean(y_scores_jp, axis=0)
y_scores_avg_tw= np.mean(y_scores_tw, axis=0)

aucs_jp = cal_aucs(y_trues, y_scores_avg_jp)
auc_jp_cis= cal_cis(y_trues, y_scores_avg_jp, targets)
auprcs_jp= cal_auprc(y_trues, y_scores_avg_jp)
auprc_jp_cis= cal_cis(y_trues, y_scores_avg_jp, targets, _auc= False, _auprc= True)
writer.writerow(['ensemble', 'AUC_JP']+ aucs_jp.tolist())
writer.writerow(['ensemble', 'AUC_JP_CI']+auc_jp_cis)
writer.writerow(['ensemble', 'AUPRC_JP']+ auprcs_jp.tolist())
writer.writerow(['ensemble', 'AUPRC_JP_CI']+auprc_jp_cis)

aucs_tw = cal_aucs(y_t, y_scores_avg_tw)
auc_tw_cis= cal_cis(y_t, y_scores_avg_tw, targets)
auprcs_tw= cal_auprc(y_t, y_scores_avg_tw)
auprc_tw_cis= cal_cis(y_t, y_scores_avg_tw, targets, _auc= False, _auprc= True)
writer.writerow(['ensemble', 'AUC_TW']+ aucs_tw.tolist())
writer.writerow(['ensemble', 'AUC_TW_CI']+auc_tw_cis)
writer.writerow(['ensemble', 'AUPRC_TW']+ auprcs_tw.tolist())
writer.writerow(['ensemble', 'AUPRC_TW_CI']+auprc_tw_cis)

csvfile.close()
csvfile2.close()

pickle_dict= {}
pickle_dict['y_trues_jp']= y_trues
pickle_dict['y_trues_tw']= y_t
pickle_dict['y_scores_jp']= y_scores_jp
pickle_dict['y_scores_tw']= y_scores_tw
pickle_dict['aucs_jp']= aucs_jp
pickle_dict['aucs_tw']= aucs_tw
pickle_dict['auc_jp_cis']= auc_jp_cis
pickle_dict['auc_tw_cis']= auc_tw_cis
pickle_dict['auprcs_jp']= auprcs_jp
pickle_dict['auprcs_tw']= auprcs_tw
pickle_dict['auprc_jp_cis']= auprc_jp_cis
pickle_dict['auprc_tw_cis']= auprc_tw_cis
with open(f'./transfer_result/transfer_result.pickle', 'wb') as f:
    pickle.dump(pickle_dict, f)