# Copyright (C) dātma, inc™ - All Rights Reserved
# Proprietary and confidential:
# Unauthorized copying of this file, via any medium is strictly prohibited
"""Generate embeddings(1,1024) for patchsize of 256x256 modelof WSI is further trained on attention model for survival risk.
Author: Narmada Naik
Date: August 27, 2023
"""

import argparse
import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
### PyTorch Imports
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, sampler
from models.mil import  MIL_Attention_FC_surv
from utils.loss import CrossEntropySurvLoss
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import neptune

PROJECT_WORKSPACE = os.environ['NEPTUNE_ENVIRON']
API_TOKEN = os.environ['API_TOKEN']
run = neptune.init_run(project=PROJECT_WORKSPACE, api_token=API_TOKEN)
def get_bag_feats(csv_file_df,label, args):
    df = pd.read_csv(csv_file_df)
    feats = df.to_numpy()
    labels = np.zeros(args.num_classes)
    labels[int(label)] = 1   
    return  feats,labels

def train(milnet, ce, train_df, args,results_dir, optimizer, cur,fold,reg_fn=None,lambda_reg=0., gc=8):
    milnet.train()
    train_loss_surv, train_loss = 0., 0.
    Tensor = torch.cuda.FloatTensor
    all_risk_scores = np.zeros((len(train_df)))
    all_censorships = np.zeros((len(train_df)))
    all_event_times = np.zeros((len(train_df)))
    idx = 0
    for batch,row in train_df.iterrows():   
        feats,label = get_bag_feats(row['wsi_feat_path'],row['label'],args)
        event_time = row['days_death']
        Y = np.array([[row['label']]])
        Y = torch.tensor(Y).to('cuda')
        c = torch.tensor([row['survival_status']]).to('cuda')
        bag_feats = Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        hazards, S, Y_hat, A,_= milnet(bag_feats)
        loss = ce(hazards=hazards, S=S, Y=Y, c=c)
        loss_value = loss.item()
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = c.item()
        all_event_times[idx] = event_time
        
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(milnet) * lambda_reg
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg
        print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(idx, loss_value + loss_reg, row['label'], float(event_time), float(risk), feats.shape[0]))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()
        idx += 1
        if (idx) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
        
    # calculate loss and error for epoch
    train_loss_surv /= len(train_df)
    train_loss /= len(train_df)
    torch.save(milnet.state_dict(), os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    run[f"train/fold_{fold}/loss"].append(train_loss)
    run[f"train/fold_{fold}/c_index"].append(c_index)
    return c_index,train_loss_surv,train_loss

def validate(milnet, ce, valid_df, results_dir, args, cur, epoch, writer,fold,  reg_fn=None,lambda_reg=0., gc=8):
    
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(valid_df)))
    all_censorships = np.zeros((len(valid_df)))
    all_event_times = np.zeros((len(valid_df)))
    idx =0
    milnet.load_state_dict(torch.load(os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur))))
    for batch,row in valid_df.iterrows():   
        feats,label = get_bag_feats(row['wsi_feat_path'],row['label'],args)
        event_time = row['days_death']
        Y = np.array([[row['label']]])
        Y = torch.tensor(Y).to('cuda')
        c = torch.tensor([row['survival_status']]).to('cuda')
        bag_feats = torch.tensor([feats]).float().to("cuda")#Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        
        with torch.no_grad():
            hazards, S, Y_hat, _, _ = milnet(bag_feats) # return hazards, S, Y_hat, A_raw, results_dict

        loss = ce(hazards=hazards, S=S, Y=Y, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(milnet) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = c.cpu().numpy()
        all_event_times[idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg
        idx +=1
    val_loss_surv /= len(valid_df)
    val_loss /= len(valid_df)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('validloss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(loss_value + loss_reg, row['label'], float(event_time), float(risk), feats.shape[0]))
    if writer:
        writer.add_scalar(f'val/fold_{fold}/loss_surv', val_loss_surv, epoch)
        writer.add_scalar(f'val/fold_{fold}/loss', val_loss, epoch)
        writer.add_scalar(f'val/fold_{fold}/c-index', c_index, epoch)
    run[f"valid/fold_{fold}/loss"].append(val_loss)
    run[f"valid/fold_{fold}/c_index"].append(c_index)
    return 

def main(args,params):  
    csv_path = "/home/ubuntu/input/WSI_100.csv"
    start = timer()
    max_seed_value = 2**32 - 1  # Maximum value for a 32-bit integer
    seeds = np.random.randint(0, max_seed_value, args.numfolds)
 
    for fold in range(0,args.numfolds):
        start = timer()
        save_writer = os.path.join(args.writer_dir,"fold"+str(fold))
        results_dir = os.path.join(args.results_dir,"fold"+str(fold))
        os.makedirs(save_writer,exist_ok=True)
        os.makedirs(results_dir,exist_ok=True)
        writer = SummaryWriter(save_writer, flush_secs=15)
        wsi_df = pd.read_csv(csv_path, low_memory=False)
        ce = CrossEntropySurvLoss()
        model = MIL_Attention_FC_surv().cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
        cur = fold
        train_df, valid_df = train_test_split(wsi_df, test_size=0.2, random_state=seeds[fold], shuffle=True)
      
        for epoch in range(1,params["n_epochs"]):
            cindex_train = train(model,ce,train_df,args,results_dir,optimizer,cur,fold)
            #if epoch%5==0:
            validate(model,ce,valid_df,results_dir,args,cur,epoch,writer,fold)
        
        end = timer()
        print('Fold %d Time: %f seconds' % (fold, end - start))
    run.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
    parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
    parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_classes', 			 type=int, default=3, help='classes based on survival months')
    parser.add_argument('--maxepoch', 			 type=int, default=20, help='maximum number of epoch')
    parser.add_argument('--feats_size', 			 type=int, default=1024, help='feature embedding from vitl size ')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
    parser.add_argument('--results_dir', 			 type=str, default="/home/ubuntu/result_survival", help='save result')
    parser.add_argument('--writer_dir', 			 type=str, default="/home/ubuntu/result_survival/tensorboard", help='save result')
    parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)') 
    parser.add_argument('--numfolds', 			 type=float, default=5, help='num of folds')
    args = parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run['survival']="AttentionNet"
    params = {
    "optimizer": "ADAM",
    "learning_rate": args.lr,
    "n_epochs": args.maxepoch,}
    run["model/parameters"] = params
    start = timer()
    results = main(args,params)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))