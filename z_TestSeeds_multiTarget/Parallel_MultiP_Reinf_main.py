from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import *
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np
import argparse

# Use REINFORCE to control arm reaches in a feedforward fashion with several multiple targets (e.g. 50)
# trained in parallel, i.e. 100 arms for each target on the same step of gradient descent
# steps
# Note: the employed shape for the parallelisation of multiple targets, each with multiple arms is: n_arms x n_targets
# so different targets for the same arm comes fist in batch shape, rather than going same target for all its arms and then move to the next target
# ln_rate = 0.00005 works best with std = 0.01325

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
parser.add_argument('--actorLr',   '-a', type=float, nargs='?',default= 5.0000e-06) # default values is based on hyper-search
parser.add_argument('--counter',   '-i', type=int, nargs='?')
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)

args = parser.parse_args()
seed = args.seed
i = args.counter
actor_ln = args.actorLr
search_hyperParam = args.hyperparam_search


# hyperparam search based on seeds: [ 42, 245, 918]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_v = seed # test seeds: [4, 418, 81,528]
torch.manual_seed(seed_v) #torch.manual_seed(1)  # FIX SEED
# first one uses ln_rate = 0.001, _2 uses ln_rate = 0.0005, _3 uses ln_rate = 0.00005
hyper_tuning = search_hyperParam

if not hyper_tuning:
    accuracy_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_Training_accur_'+str(seed_v)+'_'+str(i)+'_oneArmOptim.pt'
    episodes = 25001

else:
    accuracy_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Hyperparam_tuning/Result/Parallel_MultiReinf_Training_accur_hyperTuning_s' + str(
        seed_v) + '_' + str(i) + '_oneArm.pt'
    episodes = 12001


n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 1 #10 # n. of arms for each target
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps -1 # use last point with no zero action
vel_weight = 0.005
ln_rate = actor_ln #0.0010000000474974513 #0.00001 #0.00005 #0.0005#0.001 #0.0006
std = 0.01733333431184292 #0.01325 #0.0064
max_u = 15000
th_error = 0.025
n_target_p = 50
overall_n_arms = n_target_p * n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)

training_arm = Parall_Arm_model(tspan,x0,dev, n_arms= overall_n_arms)

# Use to randomly generate targets in front of the arm and on the max distance circumference
#target_states = training_arm.circof_random_tagrget(n_target_p)
# load target from those used for MB DPG:
target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt')


agent = Reinf_Actor_NN(std, n_arms,max_u,dev, ln_rate= ln_rate,Output_size=n_parametrised_steps*2).to(dev)
agent.apply(agent.small_weight_init)

critic = Critic_NN(n_arms,dev).to(dev)

# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_vel = []
ep_c_loss = []

training_acc = []
training_vel = []
training_crict_loss = []



for ep in range(1,episodes):


    actions = agent(target_states,False) # CAREFUL: actions in shape: n_arms x targets x n_param_steps


    t, thetas = training_arm.perform_reaching(t_step,actions)


    rwd = training_arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms) # order is tagets first and then arms

    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    agent.update(weighted_adv)

    # Learn a Q function only, not used for training
    Q_v = critic(target_states,actions.view(overall_n_arms, -1),False)

    c_loss = critic.update(weighted_adv,Q_v)
    ep_c_loss.append(c_loss.detach())

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))


    if ep % t_print == 0:

        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print
        print_c_loss = torch.mean(torch.tensor(ep_c_loss))

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("Critic loss: ", print_c_loss)
        print("BEST: ", best_acc)
        print("training accuracy: ",print_acc)
        print("training velocity: ", print_vel,"\n")

        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_crict_loss.append(print_c_loss)
        #training_actions.append(torch.mean(actions.detach(), dim=0))
        # if print_acc < th_error:
        #     break
        ep_rwd = []
        ep_vel = []
        ep_c_loss = []

if hyper_tuning:
    training_acc.append(actor_ln)


torch.save(training_acc,accuracy_file)



# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_Actor_comparison_BestParams_50arms_36.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_Training_vel_comparison_BestParams_50arms_36.pt')
# torch.save(target_states,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_TargetPoints_comparison_BestParams_50arms_36.pt')
# torch.save(training_crict_loss,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_TrainingCLoss_comparison_BestParams_50arms_36.pt')
# torch.save(critic.state_dict(),'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Results/Parallel_MultiReinf_critic_comparison_BestParams_50arms_36.pt')


# tst_actions = (agent(target_states,True)).view(n_target_p, 2, -1)
#
# test_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_target_p)
#
# _, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())
#
# rwd = torch.sqrt(test_arm.compute_rwd(thetas, target_states[:,0:1], target_states[:,1:2], f_points))
# velocity = torch.sqrt(test_arm.compute_vel(thetas, f_points))
#
# print("tst rwd: ", rwd)
# print("tst vel: ",velocity)