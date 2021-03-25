from Two_joint_arm.TD_3.TD3_FB_Actor_Critic import *
from Two_joint_arm.TD_3.TD3_FB_ArmModel import FB_Par_Arm_model
from Two_joint_arm.TD_3.TD3 import TD3
from Two_joint_arm.TD_3.Vanilla_MemoryBuffer import V_Memory_B
import torch
import numpy as np

torch.manual_seed(0) # FIX SEED

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev2 = torch.device('cpu')
#dev2 = dev



#TD_3 parameters:
n_episodes = 50000
buffer_size = 1000000
batch_size = 64 #  number of transition bataches (i.e. n_arms) sampled from buffer
start_update = 5000#0
actor_update = 2
ln_rate_c = 0.0000005 #0.0005
ln_rate_a = 0.0000005 # 0.0005
decay_upd = 0.005# 0.005
std = 0.01
action_space = 3 # two torques + decay
state_space = 7 # cosine, sine and angular vel of two torques + time
lamb = 0# 50000 # 100000 not learning anything # 1000

# Simulation parameters
n_RK_steps = 100
t_print = 1#50
n_arms = 1000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape for dynamical system
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = 15
t_range = torch.linspace(tspan[0] + t_step, tspan[1], n_RK_steps).to(dev) # time values for simulations
beta = 0.8# 0.05# 0.4
max_decay = 200
max_u = 2500

# Compute t at which t_window starts
t_window = (n_RK_steps-f_points) / n_RK_steps * tspan[-1]

# Target (x,y) endpoint, based on matlab - reach straight in front, at shoulder height
goal = (0.792,0)


# Initialise arm
env = FB_Par_Arm_model(t_step,x0,goal,t_window,dev,n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(n_arms,ln_rate = ln_rate_a).to(dev)

critic_1 = Critic_NN(lamb,dev,ln_rate=ln_rate_c).to(dev)
critic_2 = Critic_NN(lamb,dev,ln_rate=ln_rate_c).to(dev)

#Initialise Buffer:
buffer = V_Memory_B(n_arms,dev2, batch_size=batch_size,size=buffer_size)

# Initialise DPG algorithm passing all the objects
td3 = TD3(agent,critic_1,critic_2,buffer,decay_upd,n_arms,dev, actor_update= actor_update)


cum_rwd = []
cum_vel = []


# Create a mask to make sure last step isn't update with a Q-estimate
dn = torch.cat([torch.ones(n_RK_steps-1),torch.zeros(1)]).to(dev)

cum_critc_loss = []

cum_actor_loss = []

ep_actions = []

# Initialise t0 for each arm
t0 = torch.tensor([tspan[0]]).expand(n_arms,1).to(dev)

#dn = torch.ones(n_arms).to(dev)

for ep in range(1,n_episodes):


    c_state = torch.cat([env.reset(), t0],dim=1)

    t_counter = 0
    ep_rwd = []
    ep_vel = []
    step = 1

    for t in t_range:

        if ep < start_update:

            std = 0.05
            td3.actor_update = 5

        else:

            std = 0.001
            td3.actor_update = 2


        det_action = agent(c_state).detach()
        stocasticity = torch.randn(n_arms,action_space).to(dev) * std

        # Compute action to save in the buffer
        Q_action = det_action + torch.cat([stocasticity[:, 0:2], stocasticity[:, 2:].clamp(0)],dim=1)  # saved action in small range


        # Scale actions by max value to feed to arm model
        #action = torch.cat([det_action[:,0:2] * 2500, det_action[:,2:3] * 200],dim=1) + stocasticity

        action = torch.cat([Q_action[:, 0:2] * max_u, Q_action[:, 2:3] * max_decay], dim=1)

        n_state,sqrd_dist, sqrd_vel = env.step(action,t)

        n_state = torch.cat([n_state,t.expand(n_arms,1)],dim=1) # add time value to state

        rwd = sqrd_dist + (sqrd_vel * beta)

        ep_rwd.append(torch.mean(torch.sqrt(sqrd_dist)))
        ep_vel.append(torch.mean(torch.sqrt(sqrd_vel)))


        buffer.store_transition(c_state,Q_action,rwd,n_state,dn[t_counter].expand(n_arms))
        #buffer.store_transition(c_state, Q_action, rwd, n_state, dn)
        c_state = n_state

        t_counter+=1

        ep_actions.append(torch.mean(action,dim=0,keepdim=True))

        # Check if it's time to update
        if  ep > 1: #and step % 3 == 0: #t%25 == 0 and

            critic_loss1,_,actor_loss = td3.update(step)
            cum_critc_loss.append(critic_loss1.detach())
            cum_actor_loss.append(actor_loss.detach())

        step +=1

    cum_rwd.append(sum(ep_rwd[-f_points:])/(f_points))
    cum_vel.append(sum(ep_vel[-f_points:])/(f_points))



    # cum_rwd.append(sum(ep_rwd)/n_RK_steps)
    # cum_vel.append(sum(ep_vel)/n_RK_steps)

    if ep % t_print == 0:

        print("ep: ", ep)
        print("Aver final rwd: ", sum(cum_rwd)/t_print)
        print("Aver final vel: ", sum(cum_vel)/t_print)
        print("Critic loss: ", sum(cum_critc_loss)/(t_print*n_RK_steps))
        print("Actor loss", sum(cum_actor_loss)*2 / (t_print*n_RK_steps))
        max_a,_ = torch.max(torch.cat(ep_actions),dim=0)
        print("Max actions",max_a )
        min_a,_ = torch.min(torch.cat(ep_actions), dim=0)
        print("Min actions", min_a)
        #print("Actions: ",ep_actions[-1],'\n')
        cum_rwd = []
        cum_vel = []
        cum_critc_loss = []
        cum_actor_loss = []
        ep_actions = []