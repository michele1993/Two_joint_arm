from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Best so far
# Implement DPG for the FeedForward arm model, inputting the desired location to a actor NN,
# which outputs entire sequence of actions, then train a Q to predict simple MC retunr on entire
# trajectory, and differentiate through that train actor, but using two critic NNs, one target NN
# towards which policy update, which is slowly updated through polyak average towards as second critic
# NN which is constantly updated

torch.manual_seed(16)  # 16 FIX SEED

# dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')

episodes = 20000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100  # 0
n_arms = 100  #
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps  # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.005#0.05  # 0.8# 0.6
ln_rate_c = 0.005  # 0.01 #0.001# 0.005
ln_rate_a = 0.00001  # 0.000005  #0.00001 #0.000005
std = 0.01#0.01#0.01  # 0.2 #0.000015#0.02
max_u = 15000 # 10000
# actor_update = 2#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd = 100  # 50#500
th_error = 0.025
decay_upd = 0.005


print("small init critic","time_window_steps = ", time_window_steps, " ln_rate_c =", ln_rate_c, " ln_rate_a= ", ln_rate_a,
      " std =", std, " decay_upd polyak ave: ", decay_upd, " Vel W: ", vel_weight)

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)


training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps * 2 + 2
critic = Critic_NN(n_arms, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)


target_critic = Critic_NN(n_arms, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)
target_critic.load_state_dict(critic.state_dict())

#critic_1.apply(critic_1.small_weight_init)
# critic_2 = Critic_NN(input_size= c_input_s, ln_rate=ln_rate_c).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
training_actions = []


Tar_Q = torch.zeros(1)

best_acc = 50



for ep in range(1, episodes):

    det_actions = agent(target_state)  # may need converting to numpy since it's a tensor

    exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)

    Q_actions = (det_actions.view(1, 2, -1) + exploration).detach()

    actions = Q_actions * max_u

    t, thetas = training_arm.perform_reaching(t_step, actions.detach())

    rwd = training_arm.compute_rwd(thetas, x_hat, y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = (rwd + vel_weight * velocity)

    Q_v = critic(target_state, Q_actions.view(n_arms, -1), False)
    c_loss = critic.update(weighted_adv, Q_v)
    target_critic.soft_update(critic, decay_upd)

    if ep > start_a_upd:

        Tar_Q = target_critic(target_state, det_actions, True)
        agent.update(Tar_Q)


    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        #std *= 0.999

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Target Q: ", torch.mean(Tar_Q.detach()))
        print("Current Q:", torch.mean(Q_v.detach()))
        print("Mean actions: ", torch.mean(det_actions**2))


        # if print_acc < th_error:
        #     break

        ep_rwd = []
        ep_vel = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)
        #training_actions.append(torch.mean(det_actions.detach(), dim=0))
        training_confidence = []


# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/Conf3_FF_DPG_Actor_s18.pt')
# torch.save(critic_1.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/Conf3_FF_DPG_Critic_s18.pt')
# torch.save(training_acc,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Conf3_FF_DPG_Training_accur_s18.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Conf3_FF_DPG_Training_vel_s18.pt')
#torch.save(training_actions,'/home/px19783/Two_joint_arm/TD_3/FeedForward/Conf3_FF_DPG_Training_actions_1.pt')


tst_actions = (agent(target_state) * max_u).view(1, 2, -1)
test_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=1)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())



rwd = torch.sqrt(training_arm.compute_rwd(thetas, x_hat, y_hat, f_points))
velocity = torch.sqrt(training_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)