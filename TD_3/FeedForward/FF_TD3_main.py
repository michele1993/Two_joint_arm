from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')


episodes = 100000
n_RK_steps = 100
time_window_steps = 15
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 50
n_arms = 5000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps #[- time_window_steps, -1] # number of final points to average across for distance to target and velocity
vel_weight = 0.8
ln_rate_c = 0.001
ln_rate_a = 0.00005
std = 0.00033
max_u = 3000
actor_update = 5
start_a_upd  = 100


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).repeat(n_arms,1)
training_arm = FF_Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(Output_size = n_parametrised_steps *2,ln_rate = ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps *2 + 2
critic_1 = Critic_NN(input_size= c_input_s,ln_rate=ln_rate_c).to(dev)
#critic_2 = Critic_NN(input_size= c_input_s, ln_rate=ln_rate_c).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01

ep_rwd = []
ep_vel = []


for ep in range(episodes):


    det_actions = agent(target_state) # may need converting to numpy since it's a tensor

    exploration = torch.randn((n_arms,2,n_parametrised_steps)) * std
    Q_actions = det_actions.view(n_arms,2,-1) + exploration

    zero_actions = torch.zeros(n_arms, 2, time_window_steps).to(dev)
    actions = torch.cat([Q_actions * max_u, zero_actions], dim=2)

    t, thetas = training_arm.perform_reaching(t_step,actions.detach())

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)


    advantage = rwd - avr_rwd
    avr_rwd += alpha * torch.mean(advantage)

    velocity_adv = velocity - avr_vel
    avr_vel += alpha * torch.mean(velocity_adv)

    weighted_adv = advantage + vel_weight * velocity_adv #(velocity_adv + tau_adv)

    Q_v = critic_1(target_state,Q_actions.view(n_arms, -1).detach())
    critic_1.update(weighted_adv, Q_v)

    if ep > start_a_upd and ep % actor_update == 0:

        Tar_Q = - critic_1(target_state,det_actions) # want to max the advantage
        agent.update(Tar_Q)


    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        print("episode: ", ep)
        print("training accuracy: ",sum(ep_rwd)/t_print)
        print("training velocity: ", sum(ep_vel)/t_print)
        ep_rwd = []
        ep_vel = []




# tst_tspan = tspan #[0, 0.4 + (t_step* f_points)] # extend time of simulation to see if arm bounce back
# test_arm = Parall_Arm_model(tst_tspan,x0,dev,n_arms=1)
#
# test_actions = torch.unsqueeze(agent.test_actions(),0).detach()
# zero_actions = torch.zeros(1,2,time_window_steps).to(dev)
# test_actions = torch.cat([test_actions,zero_actions],dim=2)
#
# torch.save(test_actions, 'test_actions2_Viscosity_av_vel_20points.pt')
#
# agent.gaussian_convol(test_actions)
# # add some zero input for extra time
#
#
# t_t, t_y = test_arm.perform_reaching(t_step,test_actions)
#
#
# torch.save(t_y, 'test_dynamics2_Viscosity_av_vel_20points.pt')
#
#
# tst_accuracy = test_arm.compute_rwd(t_y,x_hat,y_hat,f_points)
# tst_velocity = test_arm.compute_vel(t_y, f_points)
#
#
#
# print("Test accuracy: ",torch.sqrt(tst_accuracy))
# print("Test velocity", torch.sqrt(tst_velocity))