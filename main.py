import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime

from dataset import ECG_AR
from resnet import resnet34
from utils import cal_f1s, cal_aucs, split_data, cal_auprc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/data/tewei/ExVal_JP/signal_csv', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=37, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    parser.add_argument('--jp', type=str, default= 'regroup', help='proportion of the external jp data for training')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    wandb.log({'epoch': epoch, 'loss': running_loss})
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    return running_loss
    

def evaluate(dataloader, net, args, criterion, device, epoch=0, cv_fold_num=0):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    aucs = cal_aucs(y_trues, y_scores)
    avg_auc = np.mean(aucs)
    print('AUCs:', aucs)
    print('Avg AUC: %.4f' % avg_auc)
    auprcs= cal_auprc(y_trues, y_scores)
    avg_auprc= np.mean(auprcs)
    print('AUPRCs:', auprcs)
    print('Avg AUPRC: %.4f' % avg_auprc)
    if avg_auc > args.best_metric:
        args.best_metric = avg_auc
        torch.save(net.state_dict(), f'/data/tewei/ExVal_JP/model/resnet34_{args.jp}_{database}_{cv_fold_num}_{args.leads}_best.pth')
        print('Model Saved! ', avg_auc)
    return aucs, auprcs, f1s
    # if args.phase == 'train' and avg_f1 > args.best_metric:
    #     args.best_metric = avg_f1
    #     torch.save(net.state_dict(), f'models/resnet34_{database}_{args.leads}_{args.seed}_{epoch}_{int(avg_f1*100)}.pth')
    #     print('Model Saved! ', avg_f1)
        # torch.save(net.state_dict(), args.model_path)
        


if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    config = dict(
        epochs= args.epochs,
        batch_size= args.batch_size,
        learning_rate= args.lr,
        )
    
    jp= args.jp
    # data_dir = os.path.normpath(args.data_dir)
    database = 'retrain_0814' # CHANGE HERE
    label_csv = f"/data/tewei/ExVal_JP/mixeddata_{jp}_label.csv" # CHANGE HERE
    data_dir = "/data/tewei/ExVal_JP/signal_csv/" # CHANGE HERE
    
    # if not args.model_path:
    #     args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'

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
    
    
    
    # train_folds, val_folds, test_folds = split_data(seed=args.seed)
    # print(train_folds, val_folds, test_folds)
    folds = range(2, 7)
    cv_folds = []
    # seeds = range(21, 31)
    test_folds = np.array([1])
    
    import csv
    csvfile = open('/data/tewei/ExVal_JP/results/retrain_result_{}_{}.csv'.format(jp, datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')), 'w', newline='') # CHANGE HERE
    writer = csv.writer(csvfile)
    targets = ['LVEF<55%', 'LVEF(bp)<55%', 'LVEF(bp)<50%', 'LVEF(bp)<40%', 'LVEDD>65', 'LVESDi>20', 'LVESDi>25', 'LVESDi>30', 'TRPG>30', 'TRPG>40', 'LVMi>158', 'EDVi>99', 'ESVi>37', 'ESVi>45', 'BAV', 'NYHA>=3', 'CCI>=3', 'AR_VC>=6', 'AR_VC>=8']
    
    writer.writerow(['Fold', 'Epoch', 'Target']+ targets)
    all_losses= []
    with wandb.init(project= database, config=config):
        for s, fd in enumerate(folds):
            train_folds = list(folds[:])
            train_folds.remove(fd)
            train_folds = np.random.RandomState().permutation(train_folds)
            val_folds = np.array([fd])
            print(train_folds, val_folds, test_folds)
        
            train_dataset = ECG_AR('train', data_dir, label_csv, train_folds, leads, targets)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_dataset = ECG_AR('val', data_dir, label_csv, val_folds, leads, targets)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_dataset = ECG_AR('test', data_dir, label_csv, test_folds, leads, targets)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            net = resnet34(input_channels=15, num_classes=len(targets)).to(device)
            # net = resnet34(input_channels=nleads, num_classes=19).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
            criterion = nn.BCEWithLogitsLoss()
            args.best_metric = 0
            wandb.watch(net, criterion, log_freq=10)
            all_aucs= []
            all_f1s= []
            losses= []
            for epoch in range(1, args.epochs+1):
                loss= train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
                aucs, auprcs, f1s= evaluate(val_loader, net, args, criterion, device, epoch=epoch, cv_fold_num=s)
                all_aucs.append(np.array(aucs))
                all_f1s.append(np.array(f1s))
                writer.writerow([s, epoch, 'aucs']+aucs.tolist())
                writer.writerow([s, epoch, 'auprcs']+ auprcs.tolist())
                csvfile.flush()
                losses.append(loss)
            all_losses.append(losses)
            for i in range(len(targets)):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle(f'Val_{targets[i]}')
                x = np.linspace(1, args.epochs, num= args.epochs)
                ax1.plot(x, np.array(all_aucs)[:, i])  
                ax1.set_title('fold{}_AUC')
                ax2.plot(x, np.array(all_f1s)[:, i])
                ax2.set_title('fold{}_F1')
                wandb.log({f"Val_{targets[i]}": fig})
        wandb.log({"Plots": wandb.plot.line_series(
            xs= np.linspace(1, args.epochs, num= args.epochs, dtype= np.int16).tolist(),
            ys= all_losses,
            keys= ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
            title= "Training loss",
            xname= "epochs"
        )})

            

    net.load_state_dict(torch.load(f'/data/tewei/ExVal_JP/model/resnet34_{jp}_{database}_{s}_{args.leads}_best.pth', map_location=device))
    aucs, auprcs, f1s = evaluate(test_loader, net, args, criterion, device)
    writer.writerow(['Test', 'Test', 'aucs']+aucs.tolist())
    writer.writerow(['Test', 'Test', 'auprcs']+ auprcs.tolist())
    csvfile.close()

            
