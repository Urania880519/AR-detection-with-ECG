import argparse
import os

import numpy as np
import pandas as pd
import torch
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt

from resnet import resnet34
from utils import prepare_input, prepare_input2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/data/tewei/ExVal_JP/signal_csv', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--jp', type=str, default= 'regroup', help='proportion of the external jp data for training')
    
    return parser.parse_args()


def plot_shap(ecg_data, sv_data, top_leads, patient_id, label, label_index):
    # patient-level interpretation along with raw ECG data
    leads = np.array(['I', 'II', 'III', 'II-1', 'aVR', 'aVL', 'aVF', 'II-2', 'V1', 'V2', 'V3', 'II-3', 'V4', 'V5', 'V6'])
    nleads = len(top_leads)
    if nleads == 0:
        return
    nsteps = 1238 # ecg_data.shape[1], visualize last 10 s since many patients' ECG are <=10 s
    x = range(nsteps)
    ecg_data = ecg_data[:, -nsteps:]
    sv_data = sv_data[:, -nsteps:]
    #threshold = 0.001 # set threshold to highlight features with high shap values
    #threshold= threshold #change here 1212
    fig, axs = plt.subplots(nleads, figsize=(3, 4*nleads))
    fig.suptitle(label)
    for i, lead in enumerate(top_leads):
        threshold= np.mean(abs(sv_data[lead]))
        #if i<=10:
        #    print(threshold)
        
    
        sv_upper = np.ma.masked_where(sv_data[lead] >= threshold, (ecg_data[lead]))
        sv_lower = np.ma.masked_where(sv_data[lead] < threshold, (ecg_data[lead]))
        sv_all = (ecg_data[lead])
        if nleads == 1:
            axe = axs
        else:
            axe = axs[i]
        axe.plot(x, sv_all, '#4D80E6', linewidth=0.8)
        axe.plot(x, sv_lower, '#E32636', linewidth=0.8)
        #change here
        axe.set_title(leads[lead])
        axe.set_xticks(np.arange(0, 1200, 200))
        axe.set_yticks(np.arange(-2, 2, 0.5))
        axe.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axe.xaxis.set_minor_locator(plt.MultipleLocator(40))
        axe.set_xlabel('sec', loc='right')
        axe.set_ylabel('mV')
        axe.grid(visible=True, which='major', linewidth=1.0, alpha= 0.5)
        axe.grid(visible= True, which='minor', linewidth=0.5, alpha= 0.5)
        axe.tick_params(labelleft=False)
        newlabel=[]
        for label in axe.get_xticks():
            if int(label)%1000==0:
                newlabel.append(int(label)/1000)
            else:
                newlabel.append(' ')
        axe.set_xticklabels(newlabel)
        
    plt.savefig(f'shap_0205/shap1-{patient_id}-{label_index}.png')
    plt.close(fig)


def summary_plot(svs, y_scores):
    leads = np.array(['I', 'II', 'III', 'II-1', 'aVR', 'aVL', 'aVF', 'II-2', 'V1', 'V2', 'V3', 'II-3', 'V4', 'V5', 'V6'])
    svs2 = []
    n = y_scores.shape[0]
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        sv_data = svs[label, i]
        svs2.append(np.sum(sv_data, axis=1))
    svs2 = np.vstack(svs2)
    svs_data = np.mean(svs2, axis=0)
    plt.plot(leads, svs_data)
    plt.savefig('./shap_0205/summary.png')
    plt.clf()


def plot_shap2(svs, y_scores, cmap=plt.cm.Blues):
    # population-level interpretation
    leads = np.array(['I', 'II', 'III', 'II-1', 'aVR', 'aVL', 'aVF', 'II-2', 'V1', 'V2', 'V3', 'II-3', 'V4', 'V5', 'V6'])
    n = y_scores.shape[0]
    results = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    print(svs.shape)
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        results[label].append(svs[label, i])
    ys = []
    for label in range(y_scores.shape[1]):
        result = np.array(results[label])
        y = []
        for i, _ in enumerate(leads):
            y.append(result[:,i].sum())
        y = np.array(y) / np.sum(y)
        ys.append(y)
        plt.plot(leads, y)
    ys.append(np.array(ys).mean(axis=0))
    ys = np.array(ys)
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)
    fmt = '.2f'
    xlabels = leads
    ylabels = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'] + ['AVG']
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(j, i, format(ys[i, j], fmt),
                    ha='center', va='center',
                    color='white' if ys[i, j] > thresh else 'black')
    np.set_printoptions(precision=2)
    fig.tight_layout()
    plt.savefig('./shap_0205/shap2.png')
    plt.clf()


if __name__ == '__main__':
    args = parse_args()
    # data_dir = os.path.normpath(args.data_dir)
    # database = os.path.basename(data_dir)
    # label_csv = os.path.join(data_dir, 'labels.csv')

    jp= args.jp
    database = 'retrain_0814' # CHANGE HERE
    label_csv = f"/data/tewei/ExVal_JP/mixeddata_{jp}_label.csv" # CHANGE HERE
    data_dir = "/data/tewei/ExVal_JP/signal_csv/" # CHANGE HERE
    args.model_path = f'./model/resnet34_regroup_{database}_{0}_{args.leads}_best.pth'

    
    #reference_csv = os.path.join(data_dir, 'reference.csv')
    lleads = np.array(['I', 'II', 'III', 'II-1', 'aVR', 'aVL', 'aVF', 'II-2', 'V1', 'V2', 'V3', 'II-3', 'V4', 'V5', 'V6'])
    classes = np.array(['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEF(bp)<40%', 'LVEDD>65', 'LVESDi>20', 'LVESDi>25', 'LVESDi>30', 'TRPG>30', 'TRPG>40', 'LVMi>158', 'EDVi>99', 'ESVi>37', 'ESVi>45', 'BAV', 'NYHA>=3', 'CCI>=3', 'AR_VC>=6', 'AR_VC>=8'])
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    model = resnet34(input_channels=15, num_classes=19).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    background = 10
    result_dir= f'./shap_0205/results/'
    result_path = os.path.join(result_dir, f'A{background * 2}.npy')

    # df_labels = pd.read_csv(label_csv)
    # df_reference = pd.read_csv(os.path.join(args.data_dir, 'reference.csv'))
    # df = pd.merge(df_labels, df_reference[['patient_id', 'age', 'sex', 'signal_len']], on='patient_id', how='left')

    # df = df[df['signal_len'] >= 15000]
    df = pd.read_csv(label_csv)

    patient_ids = df['patient_id'].to_numpy()
    to_explain = patient_ids[df['fold']==1]

    background_patient_ids = df.head(background)['patient_id'].to_numpy()
    background_inputs = [os.path.join(data_dir, patient_id+'.csv') for patient_id in background_patient_ids]
    background_inputs = torch.stack([torch.from_numpy(prepare_input2(input)).float() for input in background_inputs]).to(device)
    
    e = shap.GradientExplainer(model, background_inputs)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    svs = []
    y_scores = []
    for patient_id in tqdm(to_explain):
        _input = os.path.join(data_dir, patient_id+'.csv')
        inputs = torch.stack([torch.from_numpy(prepare_input2(_input)).float()]).to(device)
        y_scores.append(torch.sigmoid(model(inputs)).detach().cpu().numpy())
        sv = np.array(e.shap_values(inputs)) # (n_classes, n_samples, n_leads, n_points)
        svs.append(sv)
    svs = np.concatenate(svs, axis=1)
    y_scores = np.concatenate(y_scores, axis=0)
    #np.save(result_path, (svs, y_scores))
    #svs, y_scores = np.load(result_path, allow_pickle=True)

    # summary_plot(svs, y_scores)
    # plot_shap2(svs, y_scores)
    if not os.path.exists('./shap_0205'):
        os.mkdir('./shap_0205')

    preds = []
    
    for i, patient_id in enumerate(to_explain):
        ecg_data = prepare_input2(os.path.join(data_dir, patient_id+'.csv'))
        # label_idx = np.argmax(y_scores[i])
        for label_idx in range(19):
            top_leads_dict= {}
            sv_data = svs[label_idx, i]
            sv_data_mean = np.mean(sv_data, axis=1)
            
            top_leads = np.where(sv_data_mean > 1e-4)[0] # select top leads
            print(top_leads)
            
            #change here
            for a, b in zip(top_leads, sv_data_mean[top_leads]): 
                top_leads_dict[a]=b
            top_leads_dict= sorted(top_leads_dict.items(), key= lambda item:item[1], reverse=True)
            if len(top_leads_dict) >3:
                top_leads_dict= top_leads_dict[:3]
            top_leads_list=[i[0] for i in top_leads_dict]
            #top_leads_list= np.asarray(top_leads_list) #turn the list into array
            print(top_leads_list)
                
            preds.append(classes[label_idx])
            print(patient_id, classes[label_idx], lleads[top_leads_list])
            
            '''#change here 1212
            #top_leads_dict looks like this: [(int, shap),(int, shap),(int, shap)]
            if len(top_leads_dict)>0:
                threshold= np.mean(np.array(top_leads_dict)[:,1])
                #print(threshold)
                #print(top_leads_dict[-1][1])
            else:
                thrsehold= 0'''

            plot_shap(ecg_data, sv_data, top_leads_list, patient_id, classes[label_idx], label_idx)


