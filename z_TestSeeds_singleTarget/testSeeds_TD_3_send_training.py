import torch
import argparse
import numpy as np
from TD_3.FeedForward.TD3_train_testSeeds import TD3_train

# For hyperparam search

# NOTE: since accuracy and velocities are GPU tensors and are saved onto numpy array before
# being passed to cpu, can only open the saved file from a GPU (i.e. bluepebble) and
# not from local computer, since , I believe, numpy doesn't have map to cpu and can't open
# .npy directly with torch

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
parser.add_argument('--criticLr',    '-c', type=float, nargs='?')
parser.add_argument('--actorLr',   '-a', type=float, nargs='?')
parser.add_argument('--counter',   '-i', type=int, nargs='?')

args = parser.parse_args()
seed = args.seed
critic_ln = args.criticLr
actor_ln = args.actorLr
i = args.counter

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

torch.manual_seed(int(seed))  # re-set seeds everytime to ensure same initialisation
std = 0.0119
episodes = 15000

# redefine everything at each iteration to avoid potential memory leakages
TD3 = TD3_train(float(critic_ln), float(actor_ln), std, episodes, dev)
training_acc = TD3.train()

values = np.array(training_acc)
print(values, "\n")
np.save('/home/px19783/Two_joint_arm/TD_3/FeedForward/Results/DDPG_FF_testSeeds_s' + str(
    seed) + "_" + str(i) + '.npy', values)

