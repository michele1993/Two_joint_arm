import torch
import argparse
from MB_DPG.FeedForward.Multi_target.HyperParam_search.MultiP_MBDPG_train import MB_DPG_train

# For hyperparam search

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
parser.add_argument('--modelLr',    '-m', type=float, nargs='?')
parser.add_argument('--actorLr',   '-a', type=float, nargs='?')
parser.add_argument('--counter',   '-i', type=int, nargs='?')

args = parser.parse_args()
seed = args.seed
model_ln = args.modelLr
actor_ln = args.actorLr
i = args.counter

torch.manual_seed(int(seed))  # re-set seeds everytime to ensure same initialisation
std = 0.0124
episodes = 20001
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# redefine everything at each iteration to avoid potential memory leakages
MBDPG = MB_DPG_train(float(model_ln), float(actor_ln), std, episodes, dev)
training_acc, training_vel = MBDPG.train()

values = torch.tensor([training_acc, training_vel, model_ln, actor_ln])
print(values, "\n")
torch.save(values,'/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/HyperParam_search/Results/MultiPMB_DPG_FF_training_acc_training_s' + str(
    seed) + "_" + str(i) + '.pt')