from safety_checks.Video_arm_config import Video_arm

#from Supervised_learning.Feed_Back.FB_Decay.Spvsd_FB_Arm_model import FB_Arm_model
#from Supervised_learning.Feed_Back.FB_Decay.Spvsd_FB_Agent import Spvsd_FB_Agent

from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Arm_model import FB_L_Arm_model
from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Agent import Spvsd_FB_L_Agent

import torch
import numpy as np
import matplotlib.pyplot as plt

#Spvsd_FB_ExpTDecay_1: ep: ended quickly (converged to <= 0.0002), used fixed exp_t decay param = 9.037, time_window = 10 (with non_zero actions), ln_rate = 0.005, performed slightly more poorly than number 2
#Spvsd_FB_ExpTDecay_2: ep: 30000 (not converged to <0.0001), used fixed exp_t decay param = 9.037, time_window = 15 (with non_zero actions), ln_rate = 0.002,

#Spvsd_FB_L_RegDecay: same as above, but with time varying decay rate (outputed by NN at each step), using regularisation on norm of acceleration for entire episode, accel_weight = 0.0000005,


# Fixed Exp time decay
#test_actions = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_actions_1.pt').detach()
#dynamics = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_dynamics_1.pt').detach()
# training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_training_accuracy2.pt')
# training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_training_velocity2.pt')
# parameters = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_Decay/Results/Spvsd_FB_Decay_NNparameters_2.pt')

# Time decay learnt by the NN


training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_accuracy_cnstr2.pt')
training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_velocity_cnstr2.pt')
parameters = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_NNparameters_cnstr2.pt')


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
#window_size = 20
#decay_w = 0.6


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



# Compute final accuracy
dist_x, dist_y = test_arm.compute_rwd(dynamics,x_hat,y_hat,f_points)
tst_accuracy = torch.sqrt(dist_x + dist_y)
print(tst_accuracy[-1])

# Compute final velocity
sqrd_dx, sqrd_dy = test_arm.compute_vel(dynamics,f_points)

tst_velocity = torch.sqrt(sqrd_dx + sqrd_dy)
print(tst_velocity[-1])
#print(dynamics[100:-1,:,2:4] == dynamics[101:,:,2:4])


# Compute final acceleration
tst_acceleration = test_arm.compute_accel(tst_velocity, t_step)
print(tst_acceleration[-1])


plt.tight_layout()


fig1 = plt.figure()

ax1 = fig1.add_subplot(321)

ax1.plot(t,torch.squeeze(tst_velocity)) # take final velocity

ax1.set_title("velocity")

ax2 = fig1.add_subplot(322)

ax2.plot(t[:-1],torch.squeeze(tst_acceleration))

ax2.set_title("acceleration")


ax3 = fig1.add_subplot(323)
ax3.plot(t[:-1],test_actions[:,0,0])

ax3.set_title("u_1")

ax4 = fig1.add_subplot(324)

ax4.plot(t[:-1],test_actions[:,0,1])

ax4.set_title("u_2")

ax5 = fig1.add_subplot(325)

ax5.plot(training_steps,training_acc[1:])

ax5.set_title("Training Accuracy")

ax5.set_xlabel("x50")

ax6 = fig1.add_subplot(326)

ax6.plot(training_steps,training_vel[1:])

ax6.set_title("Training Velocity")
ax6.set_xlabel("x50")


plt.show()

video1 = Video_arm(test_arm,np.squeeze(dynamics.numpy()),t,fps=40)
video1.make_video()