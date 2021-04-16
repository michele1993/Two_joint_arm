from Supervised_learning.Feed_Forward.Supervised_Arm_Model import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.Individual_params.Supervised_agent import S_Agent
import numpy as np
import torch

#Perform supervised learning using the first attempted approach, namely, using the dynamical model as provided by Berret et al. and using the
# distance and velocity as cost function plus regularising a pair of variables

episodes = 50000
ln_rate = 1
n_RK_steps = 100
time_window = 10
n_parametrised_steps = n_RK_steps
t_print = 50
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window


# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0
#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

arm = Spvsd_Arm_model(tspan,x0,dev, n_arms=1)
agent = S_Agent(n_parametrised_steps, dev, ln_rate= ln_rate)

ep_distance = []
ep_velocity = []

velocity_weight = 0.8
regl_thor_weight = 0
regl_u_weight = 0.00005


training_accuracy= []
training_velocity = []

for ep in range(episodes):

    actions = agent.give_actions()

    t, thetas = arm.perform_reaching(t_step,actions)

    # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
    # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
    x_sqrd_dist, y_sqrd_dist = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    distance = torch.mean(x_sqrd_dist + y_sqrd_dist, dim=0, keepdim=True) # mean squared distance from target across window points for optimisation

    sqrd_dx, sqrd_dy = arm.compute_vel(thetas,f_points)
    velocity = torch.mean(sqrd_dx + sqrd_dy,dim=0,keepdim=True)

    thor1 = torch.norm(thetas[:,:,4])
    thor2 = torch.norm(thetas[:, :, 5])
    thors = thor1 + thor2

    u = torch.norm(actions[:,0:1,:]) + torch.norm(actions[:,1:2,:])

    loss = distance + (velocity * velocity_weight) + (regl_thor_weight * thors) + (regl_u_weight * u)

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
        print("Thors: ",regl_thor_weight * thors)
        print("u: ", (regl_u_weight * u), '\n')
        ep_distance = []
        ep_velocity = []



torch.save(thetas, '/Supervised_learning/Feed_Forward/Regularised/Results/Supervised_Regularised_dynamics1.pt')
torch.save(actions, '/Supervised_learning/Feed_Forward/Regularised/Results/Supervised_Regularised_actions_1.pt')
torch.save(training_accuracy,
           '/Supervised_learning/Feed_Forward/Regularised/Results/Supervised_Regularised_training_accuracy_1.pt')
torch.save(training_velocity,
           '/Supervised_learning/Feed_Forward/Regularised/Results/Supervised_Regularised_training_velocity_1.pt')



# Saftey checks of using stack(y): By using stack(), Pytorch seems to associate a gradient to all y entries, and not only to the y in the time window
# for which a gradient was explictily stored, by not using .detach(). This is because stack create one big tensor, with an overall associated gradient
# (i.e. the result of some operation for which at least one entry has a gradient, will have a gradient)
# Nevertheless, the gradients associated to entries previous to the time window don't directly affect the gradient computation of the loss,
# presumably, because, for instance, from y_0 I cannot backpropagate to ys in the time_window, so the gradient will = 0 after stack()

# This is the proof: (by optimising for distance from y_0 to target (or any other y not in the window), the actions don't change,
# although pythorch still allows you to use .backward(), but gradient must turn to zero after the backward .stack() operation since we didn't store the gradient for y_0, but only for y in time window
# thus from y0 I cannot backprop to my actions, I can only backprop to actions from the y in the time window for which the gradient was stored

#action_s.append(actions.detach().clone())
# count = []
#
# for ttt in range(t_print):
#     count.append(sum(action_s[ttt] == action_s[ttt + 1]))
#
# print(count)
# ep_distance = []
# ep_velocity = []