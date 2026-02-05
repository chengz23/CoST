import argparse
import torch
import os
from model.Det_Model import STID
from dataset.dataloader import load_dataset_Det
from utils.utils import EarlyStopping,mae,rmse,set_cpu_num,set_random_seed
from einops import rearrange
from tqdm import tqdm
from torch.optim import Adam
from utils.config import get_config

###########################################
###############  Parameters ###############
###########################################

parser = argparse.ArgumentParser(description="CoST")
parser.add_argument("--eps_model",type=str, default="STID" )
parser.add_argument("--data_name", type=str, default="SST")
parser.add_argument('--channels', type=int,default=1, help='Number of input channels')
parser.add_argument('--device', default='cuda:1', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--nsample", type=int, default=50)
parser.add_argument("--history_len", type=int, default=12)
parser.add_argument("--predict_len", type=int, default=12)
parser.add_argument("--val_batch_size",type=int,default=32)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--weight_decay",type=float,default=1e-5)
parser.add_argument("--patience",type=int,default=5)
parser.add_argument("--epochs",type=int,default=50)
args = parser.parse_args()

set_random_seed(args.seed)
set_cpu_num(5)
cfg = get_config(args)
print(cfg)

###########################################
############# Model Training  #############
###########################################

train_loader, valid_loader, test_loader, val_target_tensor,test_target_tensor,scaler=load_dataset_Det(cfg)
foldername = f"./save/deterministic_model/STID_" + args.data_name + f'_{args.history_len}_{args.predict_len}/'
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

early_stopping_mean = EarlyStopping(patience=args.epochs, verbose=True,path=(foldername+'/model.pth'))
deterministic_model=STID(cfg).to(args.device)

optimizer = Adam(deterministic_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
p1 = int(0.4 * args.epochs)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[p1], gamma=0.4)

if not os.path.exists(foldername+'/model.pth'):
    best_valid_loss = 1e10

    for epoch_no in range(args.epochs):
        avg_loss = 0
        deterministic_model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):

                observed_data = train_batch["observed_data"].to(args.device).float() # B,L,H,W
                observed_tp = train_batch["timepoints"].to(args.device).float() # B,L,2
                observed_data = rearrange(observed_data,'b l h w -> b l (h w)').unsqueeze(-1)
                x=observed_data[:,:args.history_len,:,:] # B,L,N,1
                observed_tp = observed_tp[:,:args.history_len,:]
                optimizer.zero_grad()
                predict = deterministic_model(x,observed_tp)
                loss= torch.nn.functional.mse_loss(predict,observed_data[:,args.history_len:,:,:])
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        pre_mean_list = []
        tar_mean_list = []
        deterministic_model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, valid_batch in enumerate(it, start=1):
                    observed_data = valid_batch["observed_data"].to(args.device).float()
                    observed_data = rearrange(observed_data,'b l h w -> b l (h w)').unsqueeze(-1)[:,:args.history_len,:,:]
                    observed_tp = valid_batch["timepoints"].to(args.device).float()
                    observed_tp = observed_tp[:,:args.history_len,:]

                    pre_mean = deterministic_model(observed_data,observed_tp)
                    pre_mean_list.append(pre_mean)
                    tar_mean_list.append(valid_batch["observed_data"].to(args.device).float()[:,args.history_len:,:,:])                    
                    it.set_postfix(
                        ordered_dict={
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )
            tar_mean_list = torch.cat(tar_mean_list,dim=0)
            tar_mean_list=scaler.inverse_transform(tar_mean_list.cpu())
            pre_mean_tensor = torch.cat(pre_mean_list,dim=0)
            pre_mean_tensor=scaler.inverse_transform(pre_mean_tensor.cpu())
            mae_val = mae(pre_mean_tensor,tar_mean_list)
            rmse_val= rmse(pre_mean_tensor,tar_mean_list)
            early_stopping_mean(mae_val, deterministic_model)
            print(f'Epoch {epoch_no} valid MAE: {mae_val:.4f} valid RMSE: {rmse_val:.4f}')
            if early_stopping_mean.early_stop:
                print("Early stopping! Start testing!")
                break



###########################################
############# Model Testing  ##############
###########################################
import numpy as np
deterministic_model.load_state_dict(torch.load(foldername + "model.pth"))
with torch.no_grad():
    pre_mean_list = []
    tar_mean_list = []
    cond_list = []
    deterministic_model.eval()
    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, valid_batch in enumerate(it, start=1):
            observed_data = valid_batch["observed_data"].to(args.device).float()
            observed_data = rearrange(observed_data,'b l h w -> b l (h w)').unsqueeze(-1)[:,:args.history_len,:,:]
            observed_tp = valid_batch["timepoints"].to(args.device).float()
            observed_tp = observed_tp[:,:args.history_len,:]

            pre_mean = deterministic_model(observed_data,observed_tp)
            pre_mean_list.append(pre_mean)
            tar_mean_list.append(valid_batch["observed_data"].to(args.device).float()[:,args.history_len:,:,:])   
            cond_list.append(valid_batch["observed_data"].to(args.device).float()[:,:args.history_len,:,:])  

    cond_list = torch.cat(cond_list,dim=0)
    cond_list=scaler.inverse_transform(cond_list.cpu())
    tar_mean_list = torch.cat(tar_mean_list,dim=0)
    tar_mean_list=scaler.inverse_transform(tar_mean_list.cpu())
    pre_mean_tensor = torch.cat(pre_mean_list,dim=0)
    pre_mean_tensor=scaler.inverse_transform(pre_mean_tensor.cpu())
    mae_val = mae(pre_mean_tensor,tar_mean_list)
    rmse_val= rmse(pre_mean_tensor,tar_mean_list)
    early_stopping_mean(mae_val, deterministic_model)

    print(f'Epoch {-1} test MAE: {mae_val:.4f} test RMSE: {rmse_val:.4f}')

