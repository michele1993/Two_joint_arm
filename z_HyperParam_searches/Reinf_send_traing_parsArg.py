import torch
import argparse
import numpy as np
from Vanilla_Reinf_dynamics.FeedForward.HyperParam_tuning.Reinf_train import Reinf_train

# For hyperparam search
parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
parser.add_argument('--std',    '-d', type=float, nargs='?')
parser.add_argument('--actorLr',   '-a', type=float, nargs='?')
parser.add_argument('--counter',   '-i', type=int, nargs='?')

args = parser.parse_args()
seed = args.seed
std = args.std
actor_ln = args.actorLr
i = args.counter

torch.manual_seed(int(seed))  # re-set seeds everytime to ensure same initialisation
episodes = 1001
dev = torch.device('cpu')

# redefine everything at each iteration to avoid potential memory leakages
Reinf = Reinf_train(float(std), float(actor_ln), episodes, dev)

training_acc, training_vel = Reinf.train()
values = np.array([training_acc, training_vel, std, actor_ln])

print(values, "\n")
np.save('/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/HyperParam_tuning/Results/Reinf_FF_HyperParameter_s' + str(
    seed) + "_" + str(i) + '.npy', values)