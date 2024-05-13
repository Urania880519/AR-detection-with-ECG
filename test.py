import os
import pickle
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from resnet import resnet34
from utils import cal_f1s, cal_aucs,cal_cis, split_data, find_optimal_threshold, cal_scores, cal_auprc
from dataset import ECG_AR

def scaling(X, sigma=0.1):
    # print(X.shape)
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    # print(sig.shape)
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class ECG_AR_JP(Dataset):
    def __init__(self, phase, data_dir, df, leads, targets, decimate= False):
        super(ECG_AR_JP, self).__init__()
        self.phase = phase
        self.decimate= decimate
        
        # print(df)

        #df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        # self.classes = ['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEDD>65', 'LVESDi>20', 'LVESDi>25', 'LVESDi>30', 'TRPG>30', 'LVMi>167', 'EDVi>93', 'ESVi>37', 'ESVi>45', 'BAV', 'NYHA>=3', 'AR_VC>=6', 'AR_VC>=8']
        self.classes = targets
        # self.classes = ['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEF(bp)<40%', 'LVEDD>65', 'LVESDi>25', 'AR_VC>6', 'TRPG>30', 'LVMi>med', 'EDVi>med', 'ESVi>med', 'ESVi>45' ]
        # self.classes = ['ESVi>45', 'CV Death', 'BAV', 'NYHA>=2', 'Surg', 'CCI>=2', 'SBP>130', 'DBP<70', 'PP>67' ]
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']

        ecg_data = []

        
        with open(os.path.join(self.data_dir, patient_id+'.csv'), 'r', encoding='utf-8') as csvfile:
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

        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        # ecg_data = ecg_data[-5000:, self.use_leads]
        result = np.zeros((1238, 15)) # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data

        labels = row[self.classes].to_numpy(dtype=np.float32)
        self.label_dict[patient_id] = labels
        
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

def test_scores(dataloader, net, device):
    print('Testing...')
    net.eval()
    # running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        # loss = criterion(output, labels)
        # running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)

    return y_trues, y_scores

def get_thresholds(val_loader, net, device):
    print('Finding optimal thresholds...')
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    # pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(test_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = [] 
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(cal_scores(y_true, y_pred, y_score))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    # plot_cm(y_trues, y_preds)


jp= 'regroup'
data_dir= "/raid/data/tewei/ExVal_JP/signal_csv/"
label_csv= f"/raid/data/tewei/ExVal_JP/mixeddata_{jp}_label.csv"
#JP_label= r"/home/tewei/ExVal_JP/JP_label.csv"
#JP_label= r"/home/tewei/ExVal_JP/jp_test.csv"
#JP_label= f"/raid/data/tewei/ExVal_JP/JPtest_{jp}_label.csv"
#JP_df= pd.read_csv(JP_label)
df= pd.read_csv(label_csv)
test_folds = np.array([1])
test_df= df[df['fold'].isin(test_folds)]

leads= 'all'
targets= ['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEF(bp)<40%', 'LVEDD>65', 'LVESDi>20', 'LVESDi>25', 'LVESDi>30', 'TRPG>30', 'TRPG>40', 'LVMi>158', 'EDVi>99', 'ESVi>37', 'ESVi>45', 'BAV', 'NYHA>=3', 'CCI>=3', 'AR_VC>=6', 'AR_VC>=8']
database= 'retrain_0814'
criterion = nn.BCEWithLogitsLoss()
device= 'cuda'
folds = range(2, 7)
cv_folds = []

csvfile = open(f'./results/Retrain_test_result_mixed.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['Fold', 'Target']+targets)

csvfile2 = open(f'./results/Retrain_thresholds_mixed.csv', 'w', newline='')
writer2 = csv.writer(csvfile2)
writer2.writerow(['Fold']+targets)

nets = []
y_trues_list = []
y_scores_list = []
thresholds_list = []
for s, fd in enumerate(folds):
    val_folds= np.array([fd])
    JP_dataset= ECG_AR_JP('test', data_dir, test_df, leads, targets)
    JP_loader= DataLoader(JP_dataset, batch_size= 32, shuffle= False, num_workers= 4, pin_memory= True)
    val_dataset= ECG_AR('val', data_dir, label_csv, val_folds, leads, targets)
    val_loader= DataLoader(val_dataset, batch_size=32, shuffle= False, num_workers= 4, pin_memory= True)
    net = resnet34(input_channels=15, num_classes=len(targets)).to(device)
    state_dict= torch.load(f'/raid/data/tewei/ExVal_JP/model/resnet34_{jp}_{database}_{s}_{leads}_best.pth')
    net.load_state_dict(state_dict)
    net.eval()
    nets.append(net)
    y_trues, y_scores = test_scores(JP_loader, net, device)
    y_scores_list.append(y_scores.tolist())
    y_trues_list.append(y_trues.tolist())
    aucs = cal_aucs(y_trues, y_scores)
    auprcs= cal_auprc(y_trues, y_scores)
    print(aucs)
    writer.writerow([s, 'AUC']+aucs.tolist())
    writer.writerow([s, 'AUPRC']+auprcs.tolist())
    csvfile.flush()
    

    thresholds = get_thresholds(val_loader, net, device)
    writer2.writerow([s]+thresholds)
    thresholds_list.append(thresholds)
    print('Thresholds:', thresholds)
    
    print('Results on mixed data:')
    apply_thresholds(JP_loader, net, device, thresholds)
        
y_scores_list = np.array(y_scores_list)
y_scores_avg = np.mean(y_scores_list, axis=0)

aucs = cal_aucs(y_trues, y_scores_avg)
auc_cis= cal_cis(y_trues, y_scores_avg, targets)
auprcs= cal_auprc(y_trues, y_scores)
auprc_cis= cal_cis(y_trues, y_scores_avg, targets, _auc= False, _auprc= True)
writer.writerow(['ensemble', 'AUC']+ aucs.tolist())
writer.writerow(['ensemble', 'AUC_CI']+auc_cis)
writer.writerow(['ensemble', 'AUPRC']+ auprcs.tolist())
writer.writerow(['ensemble', 'AUPRC_CI']+auprc_cis)

csvfile.close()
csvfile2.close()

pickle_dict= {}
pickle_dict['y_trues']= y_trues
pickle_dict['y_scores_avg']= y_scores
pickle_dict['aucs']= aucs
pickle_dict['auc_cis']= auc_cis
pickle_dict['auprcs']= auprcs
pickle_dict['auprc_cis']= auprc_cis
with open(f'./results/{database}_retrain_mixed.pickle', 'wb') as f:
    pickle.dump(pickle_dict, f)