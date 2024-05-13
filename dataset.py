import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
import scipy


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



import csv

class ECG_AR(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads, targets):
        super(ECG_AR, self).__init__()
        self.phase = phase
       
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        
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

        # except:
        #     print(index, patient_id, "file error", flush=True)
        
        #if self.label_dict.get(patient_id):
        #    labels = self.label_dict.get(patient_id)
        #else:
        #    labels = row[self.classes].to_numpy(dtype=np.float32)
        #    self.label_dict[patient_id] = labels
        labels = row[self.classes].to_numpy(dtype=np.float32)
        self.label_dict[patient_id] = labels
        
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)