import torch
import argparse
import numpy as np
from Vanilla_Reinf_dynamics.FeedForward.Reinf_train_testSeeds import Reinf_train

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
episodes = 15000
dev = torch.device('cpu')

# redefine everything at each iteration to avoid potential memory leakages
Reinf = Reinf_train(float(std), float(actor_ln), episodes, dev)

training_acc = Reinf.train()
values = np.array(training_acc)

print(values, "\n")
np.save('/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/Reinf_FF_testSeeds_s' + str(
    seed) + "_" + str(i) + '.npy', values)