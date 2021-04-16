from Extra_models.DDPG.DDPG_FB_Actor_Critic import *
from Extra_models.DDPG.DDPG_FB_ArmModel import FB_Par_Arm_model
from Extra_models.DDPG.DDPG_alg import DDPG
from Extra_models.DDPG.Vanilla_MemoryBuffer import V_Memory_B

import numpy as np


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dev = torch.device('cpu')

#DDPG parameters:
n_episodes = 10000
buffer_size = 50000
batch_size = 64 #  number of transition bataches (i.e. n_arms) sampled from buffer
start_update = 50
ln_rate_c = 0.002 # 0.002
ln_rate_a = 0.001 # 0.001
decay_upd = 0.0075
std = 1
beta = 0.4
action_space = 3 # two torques + decay
state_space = 8

# Simulation parameters
n_RK_steps = 100
t_print = 10
n_arms = 1#3000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape for dynamical system
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = 11
t_range = torch.linspace(tspan[0], tspan[1] - t_step, n_RK_steps).to(dev) # time values for simulations

# Compute t at which t_window starts
t_window = (n_RK_steps-f_points) / n_RK_steps * tspan[-1]

# Target (x,y) endpoint, based on matlab - reach straight in front, at shoulder height
goal = (0.792,0)


# Initialise arm
env = FB_Par_Arm_model(t_step,x0,goal,t_window,dev,n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(ln_rate = ln_rate_a,Output_size=action_space).to(dev)
critic = Critic_NN(ln_rate=ln_rate_c).to(dev)

#Initialise Buffer:
buffer = V_Memory_B(n_arms,dev,a_space=action_space,s_space = state_space, batch_size=batch_size,size=buffer_size)

# Initialise DPG algorithm passing all the objects
ddpg = DDPG(agent,critic,buffer,decay_upd,dev)


cum_rwd = []
cum_vel = []


# Create a mask to make sure last step isn't update with a Q-estimate
dn = torch.cat([torch.ones(n_RK_steps-1),torch.zeros(1)]).to(dev)

cum_critc_loss = []

ep_actions = []

for ep in range(1,n_episodes):

    c_state = env.reset()
    t_counter = 0
    ep_rwd = []
    ep_vel = []

    for t in t_range:

        det_action = agent(c_state).detach()
        stocasticity = torch.randn(n_arms,action_space).to(dev) * std
        stocasticity = torch.cat([stocasticity[:,0:2], torch.clip(stocasticity[:,2:3],0)],dim=1)
        action = det_action + stocasticity

        n_state,sqrd_dist, sqrd_vel = env.step(action,t)

        rwd = (sqrd_dist + (sqrd_vel * beta))

        ep_rwd.append(torch.mean(torch.sqrt(sqrd_dist)))
        ep_vel.append(torch.mean(torch.sqrt(sqrd_vel)))

        buffer.store_transition(c_state,action,rwd,n_state,dn[t_counter].expand(n_arms))
        c_state = n_state
        t_counter+=1

        ep_actions.append(torch.mean(action,dim=0))

        # Check if it's time to update
        if  ep > start_update: #t%25 == 0 and

            critic_loss = ddpg.update()
            cum_critc_loss.append(critic_loss.detach())


    cum_rwd.append(sum(ep_rwd[-f_points:])/(f_points))
    cum_vel.append(sum(ep_vel[-f_points:])/(f_points))


    # cum_rwd.append(sum(ep_rwd)/n_RK_steps)
    # cum_vel.append(sum(ep_vel)/n_RK_steps)

    if ep % t_print == 0:

        print("ep: ", ep)
        print("Aver final rwd: ", sum(cum_rwd)/t_print)
        print("Aver final vel: ", sum(cum_vel)/t_print)
        print("Critic loss: ", sum(cum_critc_loss)/t_print)
        print("Actions: ",ep_actions,'\n')
        cum_rwd = []
        cum_vel = []
        cum_critc_loss = []
        ep_actions = []