from Supervised_learning.Feed_Forward.Supervised_Arm_Model import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.NN_Spvsd_FF_agent import Actor_NN
import numpy as np
import torch

torch.manual_seed(1)  # FIX SEED

#Perform supervised learning using the first attempted approach, namely, using the dynamical model as provided by Berret et al. and using the
# distance and velocity as cost function, with no regularisation or decay, where the agent is a NN mapping desired states to actions.

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

episodes = 1001
n_RK_steps = 99
time_window = 0
n_parametrised_steps = n_RK_steps -time_window
t_print = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window -1
range_ln_rate = torch.linspace(0.00001,0.001,50)#0.00001
velocity_weight = 0.005
max_u = 15000
th_error = 0.01#0.025 # i.e. same accuracy as DPG at test


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


arm = Spvsd_Arm_model(tspan,x0,dev, n_arms=1)

total_acc = torch.zeros(len(range_ln_rate), 2)
total_vel = torch.zeros(len(range_ln_rate), 2)
i = 0
for ln_rate_a in range_ln_rate:

    # Initialise actor and critic
    agent = Actor_NN(max_u,dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
    agent.apply(agent.small_weight_init)

    ep_distance = []
    ep_velocity = []


    training_accuracy = None
    training_velocity = None

    for ep in range(1,episodes):

        actions = agent(target_state).view(1,2,-1)

        thetas = arm.perform_reaching(t_step,actions)

        # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
        # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
        rwd = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
        velocity = arm.compute_vel(thetas,f_points)
        loss = rwd + (velocity * velocity_weight)

        agent.update(loss)

        ep_distance.append(torch.sqrt(rwd).detach()) # mean distance to assess performance
        ep_velocity.append(torch.sqrt(velocity).detach())



        if ep % t_print == 0:

            av_acc = (sum(ep_distance)/t_print)
            av_vel = (sum(ep_velocity) / t_print)

            training_accuracy = av_acc
            training_velocity = av_vel

            print("ep: ",ep)
            print("distance: ",av_acc)
            print("velocity: ",av_vel)

            ep_distance = []
            ep_velocity = []

    total_acc[i, :] = torch.tensor([training_accuracy, ln_rate_a])
    total_vel[i, :] = torch.tensor([training_velocity, ln_rate_a])
    i += 1
    print("iteration n: ", i)



torch.save(total_acc,'/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/Results/NN_Spvsd_FF_HyperParameter_accuracy_s1.py')
torch.save(total_vel,'/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/Results/NN_Spvsd_FF_HyperParameter_velocity_s1.py')
