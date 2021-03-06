#from Supervised_learning.Feed_Forward.Decay.Spvsd_Decay_Arm_Model import Spvsd_Decay_Arm_model
from Supervised_learning.Feed_Forward.Individual_params.Decay.Spvsd_Exp_TimeDecay_Arm_Model import Spvsd_ExpDecay_Arm_model
from Supervised_learning.Feed_Forward.Individual_params.Supervised_agent import S_Agent
import numpy as np
import torch

#Perform supervised learning using the first attempted approach, namely, using the dynamical model as provided by Berret et al. and using the
# distance and velocity as cost function, with no regularisation or decay.

episodes = 100000
ln_rate= 10
n_RK_steps = 100
time_window = 10
n_parametrised_steps = n_RK_steps - time_window
t_print = 50
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window
decay_w = 9.037#10

# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0
#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

#arm = Spvsd_Decay_Arm_model(tspan,x0,dev,decay_w, n_arms=1)

arm = Spvsd_ExpDecay_Arm_model(tspan, x0, dev, decay_w, n_arms=1)
agent = S_Agent(n_parametrised_steps, dev,ln_rate= ln_rate)

ep_distance = []
ep_velocity = []

velocity_weight = 0.8

training_accuracy= []
training_velocity = []

for ep in range(episodes):

    actions = agent.give_actions()
    zero_actions = torch.zeros(1, 2, time_window).to(dev)
    actions = torch.cat([actions, zero_actions], dim=2)

    thetas = arm.perform_reaching(t_step,actions)

    # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
    # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
    x_sqrd_dist, y_sqrd_dist = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    distance = torch.mean(x_sqrd_dist + y_sqrd_dist, dim=0, keepdim=True) # mean squared distance from target across window points for optimisation

    sqrd_dx, sqrd_dy = arm.compute_vel(thetas,f_points)
    velocity = torch.mean(sqrd_dx + sqrd_dy,dim=0,keepdim=True)

    loss = distance + (velocity * velocity_weight)
    agent.update(loss)

    ep_distance.append(torch.mean(torch.sqrt(x_sqrd_dist + y_sqrd_dist)).detach()) # mean distance to assess performance
    ep_velocity.append(torch.mean(torch.sqrt(sqrd_dx + sqrd_dy)).detach())


    if ep % t_print == 0:

        av_acc = (sum(ep_distance)/t_print)
        av_vel = (sum(ep_velocity) / t_print)

        training_accuracy.append(av_acc)
        training_velocity.append(av_vel)

        print("ep: ",ep)
        print("distance: ",av_acc)
        print("velocity: ",av_vel)
        print("thor", thetas[-1,:,4:6].detach())
        print("d_thor", thetas[-1,:, 6:8].detach())
        ep_distance = []
        ep_velocity = []

        if av_acc <= 0.00005:
            break




torch.save(thetas, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Results/Supervised_ExpTDecay_dynamics_No0a.pt')
torch.save(actions, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Results/Supervised_ExpTDecay_actions_No0a.pt')
torch.save(training_accuracy,
           '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Results/Supervised_ExpTDecay_training_accuracy_No0a.pt')
torch.save(training_velocity,
           '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Results/Supervised_ExpTDecay_training_velocity_No0a.pt')

