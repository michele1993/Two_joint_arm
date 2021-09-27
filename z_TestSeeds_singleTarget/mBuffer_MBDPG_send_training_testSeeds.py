import torch
import argparse
import numpy as np
from MBDPG_MemBuffer.MBuffer_train_testSeeds import Mbuffer_MBDPG_train

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
episodes = 10001
n_arm = 1
dev = torch.device('cpu')

# redefine everything at each iteration to avoid potential memory leakages
MBDPG = Mbuffer_MBDPG_train(float(model_ln), float(actor_ln), std, episodes,n_arm, dev)
training_acc = MBDPG.train()

values = np.array(training_acc)

print(values, "\n")
np.save('/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Results/MbufferMB_DPG_FF_testSeeds_s' + str(
    seed) + "_" + str(i) + '_oneArm_Anaconda.npy', values)