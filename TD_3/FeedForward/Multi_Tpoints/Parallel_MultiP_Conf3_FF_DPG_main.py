from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.Multi_Tpoints.Multi_FF_AC import *
import torch
import numpy as np


# Best so far
# Implement DPG for the FeedForward arm model, inputting the desired location to a actor NN,
# which outputs entire sequence of actions, then train a Q to predict simple advantage on entire
# trajectory, and differentiate through that train actor, but maiting a distribution of MC returns
# and if Q value for update action is more than "some" std from this distribution, skip update to actor
# until Q estimates back to normal, same as Conf_FF implementation, just adding a region around end-point

# hyperparam search based on seeds: [ 42, 245, 918]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_v = 528 #test seeds: [4, 418, 81,528] #
torch.manual_seed(seed_v)

accuracy_file = '/home/px19783/Two_joint_arm/TD_3/FeedForward/Multi_Tpoints/Results/Parallel_MultiDPG_Training_accur_s'+str(seed_v)+'.pt'

#dev = torch.device('cpu')

episodes = 35000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100  # 0
n_arms = 10 #100  # n. of arms for each target
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps  # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05#0.05  # 0.8# 0.6
ln_rate_c = 9.99999975e-06 #0.005  # 0.01 #0.001# 0.005
ln_rate_a = 7.52500026e-04 #0.00001  # 0.000005  #0.00001 #0.000005
std = 0.0119#0.01  # 0.2 #0.000015#0.02
max_u = 15000 # 10000
# actor_update = 2#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd = 100  # 50#500
th_conf = 0.85
th_error = 0.025
n_target_p = 50
overall_n_arms = n_target_p * n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)


print("time_window_steps = ", time_window_steps, " ln_rate_c =", ln_rate_c, " ln_rate_a= ", ln_rate_a,
      " std =", std, " Confidence: ", th_conf, " Vel W: ", vel_weight)



training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=overall_n_arms)

# Use to randomly generate targets in front of the arm and on the max distance circumference
#target_states = training_arm.circof_random_tagrget(n_target_p)
target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt')

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps * 2 + 2
critic_1 = Critic_NN(n_arms, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)



ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
training_actions = []
training_confidence = []

Tar_Q = torch.zeros(1)

best_acc = 50

for ep in range(1, episodes):


    det_actions = agent(target_states)  # may need converting to numpy since it's a tensor

    # add noise to each action for each arm for each target
    exploration = (torch.randn((overall_n_arms, 2, n_parametrised_steps)) * std).to(dev)

    # need to repeat the deterministic action for each arm so can add noise to each
    Q_actions = (det_actions.repeat(n_arms,1).view(overall_n_arms, 2, n_parametrised_steps) + exploration).detach()

    actions = Q_actions * max_u

    t, thetas = training_arm.perform_reaching(t_step, actions.detach())

    rwd = training_arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    velocity = training_arm.compute_vel(thetas, f_points)

    acc_rwd = rwd.clone()

    weighted_adv = rwd + vel_weight * velocity

    Q_v = critic_1(target_states, Q_actions.view(overall_n_arms, -1), False)
    c_loss = critic_1.update(weighted_adv, Q_v)

    Tar_Q = critic_1(target_states, det_actions, True)  # want to optimise for deterministic Q

    mean_G = torch.mean(torch.mean(weighted_adv, dim=0), dim=0)
    std_G = torch.sqrt(torch.sum((mean_G - weighted_adv) ** 2) / (n_arms - 1))

    confidence = torch.mean(torch.abs((Tar_Q - mean_G) / std_G).detach()) # use mean confidence as criteria

    Tar_Q = Tar_Q[confidence <= th_conf] # only select those Q for which we have high confidence

    if ep > start_a_upd and Tar_Q.numel(): # check that Q is not empty before trying to update

        agent.update(Tar_Q)

    ep_rwd.append(torch.mean(torch.sqrt(acc_rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        print_conf = sum(training_confidence) / t_print
        #std *= 0.999

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)

        if print_acc < th_error:
            break

        ep_rwd = []
        ep_vel = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)


torch.save(training_acc,accuracy_file)
# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/Multi_Tpoints/Results/Parallel_MultiDPG_Actor_s1.pt')
# torch.save(critic_1.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/Multi_Tpoints/Results/Parallel_MultiDPG_Critic_s1.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Multi_Tpoints/Results/Parallel_MultiDPG_vel_s1.pt')

