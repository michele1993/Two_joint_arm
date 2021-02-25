from TD_3.TD3_FB_Actor_Critic import *
from TD_3.TD3_FB_ArmModel import FB_Par_Arm_model
from TD_3.TD3 import TD3
from TD_3.Vanilla_MemoryBuffer import V_Memory_B

import numpy as np


#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev2 = torch.device('cpu')
dev= dev2


#TD_3 parameters:
n_episodes = 10000
buffer_size = 50000
batch_size = 100 #  number of transition bataches (i.e. n_arms) sampled from buffer
start_update = 25
actor_update = 2
ln_rate_c = 0.00002 # 0.002
ln_rate_a = 0.00001 # 0.001
decay_upd = 0.001
std = 0.5
beta =0# 0.05# 0.4
action_space = 3 # two torques + decay
state_space = 8

# Simulation parameters
n_RK_steps = 100
t_print = 50
n_arms = 1#00
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape for dynamical system
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = 2#11
t_range = torch.linspace(tspan[0], tspan[1] - t_step, n_RK_steps).to(dev) # time values for simulations

# Compute t at which t_window starts
t_window = (n_RK_steps-f_points) / n_RK_steps * tspan[-1]

# Target (x,y) endpoint, based on matlab - reach straight in front, at shoulder height
goal = (0.792,0)


# Initialise arm
env = FB_Par_Arm_model(t_step,x0,goal,t_window,dev,n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(ln_rate = ln_rate_a,Output_size=action_space).to(dev)

critic_1 = Critic_NN(ln_rate=ln_rate_c).to(dev)
critic_2 = Critic_NN(ln_rate=ln_rate_c).to(dev)

#Initialise Buffer:
buffer = V_Memory_B(n_arms,dev2,a_space=action_space,s_space = state_space, batch_size=batch_size,size=buffer_size)

# Initialise DPG algorithm passing all the objects
td3 = TD3(agent,critic_1,critic_2,buffer,decay_upd,dev, actor_update= actor_update)


cum_rwd = []
cum_vel = []


# Create a mask to make sure last step isn't update with a Q-estimate
dn = torch.cat([torch.ones(n_RK_steps-1),torch.zeros(1)]).to(dev)

cum_critc_loss = []

cum_actor_loss = []

ep_actions = []

for ep in range(1,n_episodes):

    c_state = env.reset()
    t_counter = 0
    ep_rwd = []
    ep_vel = []
    step = 1

    for t in t_range:

        det_action = agent(c_state).detach()
        stocasticity = torch.randn(n_arms,action_space).to(dev) * std
        stocasticity = torch.cat([stocasticity[:,0:2], torch.clip(stocasticity[:,2:3],0)],dim=1)
        action = det_action + stocasticity

        n_state,sqrd_dist, sqrd_vel = env.step(action,t)

        rwd = sqrd_dist + (sqrd_vel * beta)

        ep_rwd.append(torch.mean(torch.sqrt(sqrd_dist)))
        ep_vel.append(torch.mean(torch.sqrt(sqrd_vel)))

        buffer.store_transition(c_state,action,rwd,n_state,dn[t_counter].expand(n_arms))
        c_state = n_state
        t_counter+=1

        ep_actions.append(torch.mean(action,dim=0))

        # Check if it's time to update
        if  ep > start_update: #t%25 == 0 and

            critic_loss1,_,actor_loss = td3.update(step)
            cum_critc_loss.append(critic_loss1.detach())
            cum_actor_loss.append(actor_loss)

        step +=1

    cum_rwd.append(sum(ep_rwd[-f_points:])/(f_points))
    cum_vel.append(sum(ep_vel[-f_points:])/(f_points))



    # cum_rwd.append(sum(ep_rwd)/n_RK_steps)
    # cum_vel.append(sum(ep_vel)/n_RK_steps)

    if ep % t_print == 0:

        print("ep: ", ep)
        print("Aver final rwd: ", sum(cum_rwd)/t_print)
        print("Aver final vel: ", sum(cum_vel)/t_print)
        print("Critic loss: ", sum(cum_critc_loss)/t_print)
        print("Actor loss", sum(cum_actor_loss)*2 / t_print)
        print("Actions", torch.norm(action))
        #print("Actions: ",ep_actions[-1],'\n')
        cum_rwd = []
        cum_vel = []
        cum_critc_loss = []
        cum_actor_loss = []
        ep_actions = []