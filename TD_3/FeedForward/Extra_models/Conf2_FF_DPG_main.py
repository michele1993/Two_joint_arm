from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Implement DPG for the FeedForward arm model, inputting the desired location to a actor NN,
# which outputs entire sequence of actions, then train a Q to predict simple advantage on entire
# trajectory, and differentiate through that train actor, but computing a distribution of MC returns
# and if average Q values for exploratory actions is more than some std from mean of G returns from the same exploratory actions
# delay actor update

torch.manual_seed(0) # FIX SEED

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')



episodes = 120000
n_RK_steps = 100
time_window_steps = 1 #15
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100 #0
n_arms = 100 #
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps #[- time_window_steps, -1] # number of final points to average across for distance to target and velocity
vel_weight = 0.8 # 0.6
ln_rate_c = 0.005#0.01 #0.001# 0.005
ln_rate_a = 0.000001  #0.00001 #0.000005
std = 0.01 #0.2 #0.000015#0.02
max_u = 15000#10000
#actor_update = 2#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd  = 100#50#500

print("time_window_steps = ",time_window_steps," ln_rate_c =", ln_rate_c," ln_rate_a= ", ln_rate_a," std =", std)


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
training_confidence = []

Tar_Q = torch.zeros(1)
update = True



for ep in range(episodes):


    det_actions = agent(target_state) # may need converting to numpy since it's a tensor

    exploration = (torch.randn((n_arms,2,n_parametrised_steps)) * std).to(dev)

    Q_actions = (det_actions.view(1,2,-1) + exploration).detach()

    zero_actions = torch.zeros(n_arms, 2, time_window_steps).to(dev)

    actions = torch.cat([Q_actions * max_u, zero_actions], dim=2)
    #actions = torch.cat([Q_actions, zero_actions], dim=2)

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

    Q_check = torch.mean(Q_v)

    mean_G = torch.mean(torch.mean(weighted_adv,dim=0),dim=0)
    std_G = torch.sqrt(torch.sum((mean_G - weighted_adv)**2)/ (n_arms -1))

    confidence = torch.abs((Q_check - mean_G) / std_G).detach()
    training_confidence.append(confidence)

    if ep > start_a_upd and confidence <= 0.085:

        Tar_Q = critic_1(target_state, det_actions, True)  # want to max the advantage
        agent.update(Tar_Q)


    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        #update = True
        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print
        print_conf = sum(training_confidence)/t_print
        std *= 0.9995#9

        # if print_conf > 1:
        #     update = False

        print("episode: ", ep)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Target Q: ", torch.mean(Tar_Q.detach()))
        print("Current Q:", torch.mean(Q_v.detach()))
        print("Confidence: ", print_conf,"\n")
        print("actor std: ", std)
        print("mean G ",mean_G)
        print("Target Q: ",Tar_Q)
        print("std G ",std_G,'\n')



        ep_rwd = []
        ep_vel = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_actions.append(torch.mean(det_actions.detach(),dim=0))
        training_confidence = []



# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Actor_2.pt')
# torch.save(critic_1.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Critic_2.pt')
# torch.save(training_acc,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_accur_2.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_vel_2.pt')
# torch.save(training_actions,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_actions_2.pt')