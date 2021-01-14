from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Arm_model import FB_L_Arm_model
from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Agent import Spvsd_FB_L_Agent

import torch
import numpy as np
import matplotlib.pyplot as plt



# Time decay learnt by the NN


training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_accuracy_cnstr1.pt')
training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_velocity_cnstr1.pt')
parameters = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_NNparameters_cnstr1.pt')


dev = torch.device('cpu')

agent = Spvsd_FB_L_Agent(dev)
agent.load_state_dict(parameters)

longer= False

if longer:
    n_RK_steps = 200
    tspan = [0, 0.8]
else:
    n_RK_steps = 100 #150
    tspan = [0, 0.4] #[0, 0.6]

x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] #initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps


# Target endpoint, based on matlab - reach straight in front, atshoulder height
x_hat = 0.792
y_hat = 0

t = np.linspace(tspan[0],tspan[1],n_RK_steps+1) # np.linspace(0,0.6,151)
training_steps = np.linspace(1,len(training_acc)-1,len(training_acc)-1)
decay_w = 9.037

# Use linear time increase on exponential decay
test_arm = FB_L_Arm_model(tspan,x0,dev,n_arms=1)

dynamics, test_actions,_ = test_arm.perform_reaching(t_step,agent)
dynamics = dynamics.detach()
test_actions = test_actions.detach()


f_points = - dynamics.size()[0]


# Compute variables to check constraints
tor = torch.squeeze(dynamics[:,:,4:6])
tor_velocity = torch.squeeze(dynamics[:,:,6:8])
angular_velocity = torch.squeeze(dynamics[:,:,2:4])



plt.tight_layout()


fig1 = plt.figure()

ax1 = fig1.add_subplot(421)

ax1.plot(t,torch.squeeze(angular_velocity[:,0])) # take final velocity

ax1.set_title("t_1 velocity")

ax2 = fig1.add_subplot(422)

ax2.plot(t,torch.squeeze(angular_velocity[:,1]))

ax2.set_title("t_2 velocity")


ax3 = fig1.add_subplot(423)
ax3.plot(t[:-1],test_actions[:,0,0])

ax3.set_title("u_1")

ax4 = fig1.add_subplot(424)

ax4.plot(t[:-1],test_actions[:,0,1])

ax4.set_title("u_2")

ax5 = fig1.add_subplot(425)

ax5.plot(t,tor[:,0])

ax5.set_title("Thor 1")


ax6 = fig1.add_subplot(426)

ax6.plot(t,tor[:,1])

ax6.set_title("Thor 2")


ax7 = fig1.add_subplot(427)

ax7.plot(t,tor_velocity[:,0])

ax7.set_title("Thor Vel 1")


ax8 = fig1.add_subplot(428)

ax8.plot(t,tor_velocity[:,1])

ax8.set_title("Thor Vel 2")


plt.show()