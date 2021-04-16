from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Implement DPG for the FeedForward arm model, inputting the desired location to a actor NN,
# which outputs entire sequence of actions, then train a Q to predict simple advantage on entire
# trajectory, and differentiate through that train actor, also regularising how far the new Q value
# can be from the actual return

torch.manual_seed(0) # FIX SEED

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')

episodes = 120000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 100#100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -time_window_steps -1 #[- time_window_steps, -1] # number of final points to average across for distance to target and velocity
vel_weight = 0.4#0.6#0.8
ln_rate_c = 0.01 # 0.005 #0.001
ln_rate_a = 0.00001 #0.00001 #0.000005
std = 0.01#0.01 #0.01 #0.0001
max_u = 15000
actor_update = 1#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd  = 500
Q_reg_weight = 50

print("time_window_steps = ", time_window_steps, " ln_rate_c =", ln_rate_c, " ln_rate_a= ", ln_rate_a,
      " No decay std =", std, " Q_reg_w: ", Q_reg_weight, " Vel W: ", vel_weight)

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev) #.repeat(n_arms,1).to(dev)
training_arm = FF_Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(dev,Output_size = n_parametrised_steps *2,ln_rate = ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps *2 + 2
critic_1 = Critic_NN(n_arms, dev,input_size= c_input_s,ln_rate=ln_rate_c).to(dev)
#critic_2 = Critic_NN(input_size= c_input_s, ln_rate=ln_rate_c).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
training_actions = []

Tar_Q = torch.zeros(1)

wei_reg_Q = torch.zeros(1)

best_acc = 50

for ep in range(1,episodes):


    det_actions = agent(target_state) # may need converting to numpy since it's a tensor

    exploration = torch.cat([torch.zeros((1,2,n_parametrised_steps)) ,torch.randn((n_arms-1,2,n_parametrised_steps)) * std],dim=0).to(dev)

    Q_actions = (det_actions.view(1,2,-1) + exploration).detach()

    #Q_actions = agent.gaussian_convol(Q_actions) # apply convolution to undertaken actions

    # zero_actions = torch.zeros(n_arms, 2, time_window_steps).to(dev)
    # actions = torch.cat([Q_actions * max_u, zero_actions], dim=2)

    actions = Q_actions * max_u

    t, thetas = training_arm.perform_reaching(t_step,actions.detach())

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    advantage = rwd - avr_rwd
    avr_rwd += alpha * torch.mean(advantage)

    velocity_adv = velocity - avr_vel
    avr_vel += alpha * torch.mean(velocity_adv)

    #weighted_adv = advantage + vel_weight * velocity_adv #(velocity_adv + tau_adv)
    weighted_adv = rwd + vel_weight * velocity

    Q_v = critic_1(target_state,Q_actions.view(n_arms, -1), False)
    c_loss = critic_1.update(weighted_adv, Q_v)


    if ep > start_a_upd and ep % actor_update == 0:

        Tar_Q = critic_1(target_state, det_actions, True)  # want to max the advantage

        # avr_G = torch.mean(torch.sum(weighted_adv,dim=0),dim=0,keepdim=True)
        ts_adv = weighted_adv[0,0]
        wei_reg_Q = (Tar_Q - ts_adv)**2 * Q_reg_weight

        agent.update(Tar_Q + wei_reg_Q)

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        #std *= 0.999
        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Target Q: ", torch.mean(Tar_Q.detach()))
        print("Target error", wei_reg_Q)
        print("Current Q:", torch.mean(Q_v.detach()),"\n")
        ep_rwd = []
        ep_vel = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_actions.append(torch.mean(det_actions.detach(),dim=0))



# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/RegFF_DPG_Actor_2.pt')
# torch.save(critic_1.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/RegFF_DPG_Critic_2.pt')
# torch.save(training_acc,'/home/px19783/Two_joint_arm/TD_3/FeedForward/RegFF_DPG_Training_accur_2.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/RegFF_DPG_Training_vel_2.pt')
# torch.save(training_actions,'/home/px19783/Two_joint_arm/TD_3/FeedForward/RegFF_DPG_Training_actions_2.pt')