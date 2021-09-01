from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')

episodes = 5001
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100  # 0
n_arms = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps  # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.005 #0.005
#ln_rate_c = 0.005  # 0.01 #0.001# 0.005
range_ln_rate = torch.linspace(0.00001,0.001,10)#0.00001

#range_std = torch.linspace(0.001,0.05,10)
std = 0.0119

max_u = 15000 # 10000
# actor_update = 2#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd = 100  # 50#500
th_conf = 0.85#8.25#0.85
th_error = 0.025


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)


alpha = 0.01

Tar_Q = torch.zeros(1)

best_acc = 50

update = True

total_acc = torch.zeros(len(range_ln_rate)*len(range_ln_rate), 3)
total_vel = torch.zeros(len(range_ln_rate)*len(range_ln_rate), 3)

i = 0

for ln_rate_a in range_ln_rate:
    for ln_rate_c  in range_ln_rate:
    #for std in range_std:

        training_acc = None
        training_vel = None
        ep_rwd = []
        ep_vel = []

        torch.manual_seed(1)

        training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)

    # Initialise actor and critic
        agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
        agent.apply(agent.small_weight_init)

        c_input_s = n_parametrised_steps * 2 + 2
        critic_1 = Critic_NN(n_arms, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)

        for ep in range(1, episodes):

            det_actions = agent(target_state)  # may need converting to numpy since it's a tensor

            exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)

            Q_actions = (det_actions.view(1, 2, -1) + exploration).detach()

            actions = Q_actions * max_u

            t, thetas = training_arm.perform_reaching(t_step, actions.detach())

            rwd = training_arm.compute_rwd(thetas, x_hat, y_hat, f_points)
            velocity = training_arm.compute_vel(thetas, f_points)

            acc_rwd = rwd.clone()

            weighted_adv = (rwd + vel_weight * velocity)

            Q_v = critic_1(target_state, Q_actions.view(n_arms, -1), False)
            c_loss = critic_1.update(weighted_adv, Q_v)

            Tar_Q = critic_1(target_state, det_actions, True)  # want to max the advantage

            # UNCOMMENT for multiple arms, best version
            mean_G = torch.mean(torch.mean(weighted_adv, dim=0), dim=0)
            std_G = torch.sqrt(torch.sum((mean_G - weighted_adv) ** 2) / (n_arms - 1))
            confidence = torch.abs((Tar_Q - mean_G) / std_G).detach()

            if ep > start_a_upd and confidence <= th_conf: # ep % 2 == 0

                agent.update(Tar_Q)

            ep_rwd.append(torch.mean(torch.sqrt(acc_rwd)))
            ep_vel.append(torch.mean(torch.sqrt(velocity)))

            if ep % t_print == 0:

                print_acc = sum(ep_rwd) / t_print
                print_vel = sum(ep_vel) / t_print

                print("episode: ", ep)
                print("training accuracy: ", print_acc)
                print("training velocity: ", print_vel)
                training_acc = print_acc
                training_vel = print_vel
                ep_rwd = []
                ep_vel = []

        #total_acc[i,:] = torch.tensor([training_acc, std, ln_rate_a])
        #total_vel[i,:] = torch.tensor([training_vel, std, ln_rate_a])

        total_acc[i,:] = torch.tensor([training_acc, ln_rate_c, ln_rate_a])
        total_vel[i,:] = torch.tensor([training_vel, ln_rate_c, ln_rate_a])
        i +=1
        print("iteration n: ", i)



torch.save(total_acc,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Results/Conf3_FF_DPG_HyperParam_accur_s1_ln_rates.pt')
torch.save(total_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Results/Conf3_FF_DPG_HyperParam_vel_s1_ln_rates.pt')