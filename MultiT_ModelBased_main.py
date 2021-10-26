from Model_based.MB_Arm_model import MB_FF_Arm_model
from Model_based.MB_NN_Agent import MB_Actor_NN
from Model_based.MultiTarget.MultiT_ModelBased_alg import MB_alg
import torch
import numpy as np

import argparse

# trial inputs: -s 0 -m 0.0034000000450760126 -a 0.0001 -i 1

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?')
parser.add_argument('--counter',   '-i', type=int, nargs='?')


args = parser.parse_args()
seed_v = args.seed
i = args.counter

dev = torch.device("cpu")

acc_file = '/home/px19783/Two_joint_arm/Model_based/MultiTarget/Result/MultiTModelBasedAccuracy_s'+str(seed_v)+"_"+str(i)+'.pt'
modelUp_file = '/home/px19783/Two_joint_arm/Model_based/MultiTarget/Result/MultiTModelBasedModUpdates_s'+str(seed_v)+"_"+str(i)+'.pt'
actorUp_file = '/home/px19783/Two_joint_arm/Model_based/MultiTarget/Result/MultiTModelBasedActUpdates_s'+str(seed_v)+"_"+str(i)+'.pt'
#vel_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Model_based/MultiTarget/Result/MultiTModelBasedVelocity_s'+str(seed_v)+'_uniform.pt'


Overall_episodes = 500
Model_episodes = 100
n_RK_steps = 99
time_window = 0
n_parametrised_steps = n_RK_steps -time_window
t_print = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window -1
ln_rate_a = 0.01 #0.01 works best #works well: 0.001 # 0.00001
velocity_weight = 0.005
max_u = 15000
th_error = 0.01#0.025
n_arms = 1#10 #100
Model_ln_rate = 0.005 #0.1 works best  #works well0.05#0.01 #0.08
std = 0.01
n_target_p = 50
n_overall_arms = n_target_p * n_arms



target_arm = MB_FF_Arm_model(False,tspan,x0,dev, n_arms=n_overall_arms)
estimated_arm = MB_FF_Arm_model(True,tspan,x0,dev, n_arms=n_overall_arms,ln_rate = Model_ln_rate)

agent = MB_Actor_NN(max_u,dev,Output_size= n_parametrised_steps*2, ln_rate= ln_rate_a)
agent.apply(agent.small_weight_init)

MB_alg = MB_alg(estimated_arm,agent ,t_step, n_parametrised_steps,velocity_weight, th_error, n_arms)


#target_states = target_arm.circof_random_tagrget(n_target_p)
target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt', map_location=torch.device('cpu'))


ep_distance = []
ep_model_ups = []
ep_actor_ups = []


for ep in range(1,Overall_episodes):


    actions = agent(target_states).view(n_target_p,2,n_parametrised_steps).detach()

    # exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)
    # actions = (det_actions + exploration)


    target_ths = target_arm.perform_reaching(t_step, actions)

    rwd = target_arm.multiP_compute_rwd(target_ths,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    velocity = target_arm.compute_vel(target_ths, f_points)

    acc = torch.mean(torch.sqrt(rwd))
    ep_distance.append(acc)

    # if torch.mean(torch.sqrt(rwd)) > th_error:

    print("Rollouts: ", ep)
    print("Overall Accuracy: ", acc)
    print("Overall Velocity: ", torch.mean(torch.sqrt(velocity)), "\n")

    ep_modelUp = MB_alg.update_model(actions.detach(), target_ths, target_arm)
    ep_model_ups.append(ep_modelUp)

    print("Eps to update model: ", ep_modelUp)

    ep_actorUp = MB_alg.update_actor(target_states,f_points)
    ep_actor_ups.append(ep_actorUp)

torch.save(ep_distance,acc_file)
torch.save(ep_model_ups, modelUp_file)
torch.save(ep_actor_ups, actorUp_file)