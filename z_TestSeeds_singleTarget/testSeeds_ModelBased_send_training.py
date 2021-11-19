import torch
import argparse
import numpy as np
from Model_based.TestSeeds.ModelBased_sendTraining import ModelBased_train


parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
parser.add_argument('--counter',   '-i', type=int, nargs='?')

args = parser.parse_args()
seed = args.seed
model_ln = 0.005
actor_ln = 0.01
i = args.counter

torch.manual_seed(int(seed))  # re-set seeds everytime to ensure same initialisation

episodes = 500
dev = torch.device('cpu')

# redefine everything at each iteration to avoid potential memory leakages
ModelBased = ModelBased_train(float(model_ln), float(actor_ln), episodes, dev)
ep_distance, ep_model_ups, ep_actor_ups = ModelBased.train()

value_acc = np.array(ep_distance)
value_model = np.array(ep_model_ups)
value_actor = np.array(ep_actor_ups)


print(value_acc, "\n")

np.save('/home/px19783/Two_joint_arm/Model_based/TestSeeds/Results/ModelBased_testSeeds_s' + str(
    seed) + "_" + str(i) + '.npy', value_acc)
np.save('/home/px19783/Two_joint_arm/Model_based/TestSeeds/Results/ModelBased_ModelEps_testSeeds_s' + str(
    seed) + "_" + str(i) + '.npy', value_model)
np.save('/home/px19783/Two_joint_arm/Model_based/TestSeeds/Results/ModelBased_ActorEps_testSeeds_s' + str(
    seed) + "_" + str(i) + '.npy', value_actor)