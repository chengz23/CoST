import argparse
import torch
import os
from model.Det_Model import STID
from model.Diffussion import Diff_Forecasting
from dataset.dataloader import load_dataset_Diff as load_dataset
from utils.utils import train,evaluate,set_random_seed,set_cpu_num
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
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--nsample", type=int, default=50)
parser.add_argument("--history_len", type=int, default=12)
parser.add_argument("--predict_len", type=int, default=12)
parser.add_argument("--val_batch_size",type=int,default=24)
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
######## Load Deterministic Model  ########
###########################################

deterministic_model = STID(cfg)
deterministic_model_foldername = "./save/deterministic_model/STID_" + args.data_name + f'_{args.history_len}_{args.predict_len}/'
deterministic_model.load_state_dict(torch.load(deterministic_model_foldername + "model.pth"))
print('Successfully load deterministic model model !!!')
deterministic_model.to(args.device)


###########################################
###### Probabilistic Model Training  ######
###########################################

train_loader, valid_loader, test_loader, val_target_tensor,test_target_tensor,scaler=load_dataset(cfg)
model = Diff_Forecasting(cfg)
model.to(args.device)
foldername = f"./save/probabilistic_model/eps_model_{args.eps_model}_deterministic_model_STID_" + args.data_name + f'_{args.history_len}_{args.predict_len}/'
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

train(
    deterministic_model,
    model,
    cfg,
    train_loader,
    scaler=scaler,
    valid_target=val_target_tensor,
    valid_loader=valid_loader,
    foldername=foldername,
)


###########################################
############# Model Testing  ##############
###########################################
model.load_state_dict(torch.load(foldername+ "/model.pth"))
print('test dataset:')
_,(predict,target)=evaluate(deterministic_model,args.eps_model,args,model, test_loader,test_target_tensor,scaler,nsample=args.nsample,test=1)