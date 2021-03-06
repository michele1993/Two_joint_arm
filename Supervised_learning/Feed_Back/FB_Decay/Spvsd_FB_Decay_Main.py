import torch
import numpy as np
from Supervised_learning.Feed_Back.FB_Decay.Spvsd_FB_Agent import Spvsd_FB_Agent
from Supervised_learning.Feed_Back.FB_Decay.Spvsd_FB_Arm_model import FB_Arm_model

episodes = 30000
ln_rate = 0.002 # may need a lower one, it's oscillating
n_RK_steps = 100
time_window = 15
t_print = 50
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window
decay_weight = 9.037

# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0
#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

arm = FB_Arm_model(tspan,x0,dev,decay_weight, n_arms=1)
agent = Spvsd_FB_Agent(dev,ln_rate= ln_rate)

ep_distance = []
ep_velocity = []

velocity_weight = 0.8

training_accuracy= []
training_velocity = []

for ep in range(episodes):

    thetas, u_s = arm.perform_reaching(t_step,agent)

    # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
    # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
    x_sqrd_dist, y_sqrd_dist = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    sqrd_distance = torch.mean(x_sqrd_dist + y_sqrd_dist, dim=0, keepdim=True) # mean squared distance from target across window points for optimisation

    sqrd_dx, sqrd_dy = arm.compute_vel(thetas,f_points)
    velocity = torch.mean(sqrd_dx + sqrd_dy,dim=0,keepdim=True)


    loss = sqrd_distance + (velocity * velocity_weight) # sum of squared distance and weighted squared velocity

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
        ep_distance = []
        ep_velocity = []

        if av_acc < 0.0001:
            break



torch.save(thetas, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_dynamics_2.pt')
torch.save(u_s, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_actions_2.pt')
torch.save(agent.state_dict(), '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_NNparameters_2.pt')
torch.save(training_accuracy,
           '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_training_accuracy2.pt')
torch.save(training_velocity,
                      '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_training_velocity2.pt')