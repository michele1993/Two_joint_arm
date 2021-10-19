from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.Multi_Tpoints.Multi_FF_AC import *
from TD_3.FeedForward.Buffered_DDPG.Multi_target.MultiP_VanReplay_buffer import V_Memory_B
import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?')
# Default values represent best values from hyperparam search:
parser.add_argument('--criticLr',    '-c', type=float, nargs='?', default = 5e-05) # default value based on optimal of hyperparam search
parser.add_argument('--actorLr',   '-a', type=float, nargs='?' , default = 1e-05) # default value based on optimal of hyperparam search
parser.add_argument('--counter',   '-i', type=int, nargs='?')
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)

args = parser.parse_args()
seed = args.seed
i = args.counter
critic_ln = args.criticLr
actor_ln = args.actorLr
search_hyperParam = args.hyperparam_search


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)



hyper_tuning = search_hyperParam

if not hyper_tuning:

    accuracy_file = '/home/px19783/Two_joint_arm/TD_3/FeedForward/Buffered_DDPG/Multi_target/Result/MultiDDPG_training_acc_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim.pt'
    episodes = 24001
else:

    accuracy_file = '/home/px19783/Two_joint_arm/TD_3/FeedForward/Buffered_DDPG/Multi_target/Hyperparam_tuning/Result/MultiDDPG_Training_accur_hyperTuning_s'+str(seed)+ '_' + str(i) +'_oneArm.pt'
    episodes = 12001


n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100  # 0
n_arms = 1 #100  # n. of arms for each target
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps  # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05
ln_rate_c = critic_ln
ln_rate_a = actor_ln
std = 0.0119
max_u = 15000
start_a_upd = 100
n_target_p = 50
overall_n_arms = n_target_p * n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)
a_size = n_parametrised_steps * 2
batch_size = 3000
buffer_size = 25000 #



training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=overall_n_arms)

# Use to randomly generate targets in front of the arm and on the max distance circumference


target_states = training_arm.circof_random_tagrget(n_target_p)

target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt')



# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps * 2 + 2
critic_1 = Critic_NN(n_arms, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)

M_buffer = V_Memory_B(a_size,n_arms,dev,batch_size = batch_size, n_targets=n_target_p,size=buffer_size)

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
Q_cost = []


best_acc = 50

for ep in range(1, episodes):


    det_actions = agent(target_states)  # may need converting to numpy since it's a tensor

    # add noise to each action for each arm for each target
    exploration = (torch.randn((overall_n_arms,  a_size)) * std).to(dev)

    # need to repeat the deterministic action for each arm so can add noise to each
    Q_actions = (det_actions.repeat(n_arms,1).view(overall_n_arms, a_size) + exploration).detach()

    actions = Q_actions.view(-1, 2, n_parametrised_steps) * max_u

    t, thetas = training_arm.perform_reaching(t_step, actions.detach())

    rwd = training_arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    velocity = training_arm.compute_vel(thetas, f_points)

    acc_rwd = rwd.clone()

    weighted_adv = rwd + vel_weight * velocity

    # Store transition
    M_buffer.store_transition(target_states, Q_actions, weighted_adv)

    # Sample from buffer

    spl_t_state, spl_a, spl_rwd = M_buffer.sample_transition()

    Q_v = critic_1(spl_t_state, spl_a, True) # use True since by sampling from buffer, already repeated the target_states

    c_loss = critic_1.update(spl_rwd, Q_v)
    Q_cost.append(c_loss.detach())

    if ep > start_a_upd: # check that Q is not empty before trying to update

        Tar_Q = critic_1(target_states, det_actions, True) # use True since by sampling from buffer, already repeated the target_states
        agent.update(Tar_Q)

    ep_rwd.append(torch.mean(torch.sqrt(acc_rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        print_Qcost = sum(Q_cost) / t_print
        #std *= 0.999

        if print_acc < best_acc:
            best_acc = print_acc

        print("\n episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Q loss: ", print_Qcost)


        ep_rwd = []
        ep_vel = []
        Q_cost = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)


if not hyper_tuning:
    torch.save(training_acc,accuracy_file)

else:
    training_acc.append(ln_rate_c)
    training_acc.append(ln_rate_a)
    torch.save(training_acc,accuracy_file)
