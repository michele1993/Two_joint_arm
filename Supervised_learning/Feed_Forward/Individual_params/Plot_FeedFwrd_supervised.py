from Supervised_learning.Feed_Forward.Individual_params.Decay.Spvsd_Exp_TimeDecay_Arm_Model import Spvsd_ExpDecay_Arm_model
from safety_checks.Video_arm_config import Video_arm
import torch
import numpy as np
import matplotlib.pyplot as plt

# Supervised_Basic_2: 50000 eps, time window: 10 with no zero_actions, optimised accuracy & velocity,vel_weight = 0.8,ln_rate = 1 ,Viscosity * 15

# Supervised_Decay_Correct_1 : 100000 eps, time window:10 with no zero_actions, decay_w = 0.6, ptimised accuracy & velocity,vel_weight = 0.8,ln_rate = 10 ,Viscosity * 15

# Supervised_L_Decay_Correct_1: same as above, but learning decay rate with learning_rate = 1e-3 and clipping it between 0 and 1

# Supervised_Large_Decay_1: Ended at ep 10600, decay_w = 10 (fixed), the rest was the same as above

# Supervised_LinearTDecay_1 : Ended at 8000eps ish, reduce decay_w in time linearly with slope = 5, the rest same as above

# Supervsed_ExpTDecay_1:  used exp param = 9.037 and zero actions during time window for training  (for time window =10)

# Spvsd_L_ExpTDecay_1 : work best, learnt exp_param for time-dependent decay (decay = exp(t * exp_param)) and used zero actions for time window (time window =10), the rest same as
# above, with learning rate for exp_param = 1e-1

# Learning decay data:
#test_actions = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_Correct_L_Decay_actions_1.pt',map_location=torch.device('cpu')).detach()
#decay_w  = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_Correct_L_DecayParameter_1.pt',map_location=torch.device('cpu')).detach()
#training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_Correct_L_training_accuracy_1.pt',map_location=torch.device('cpu'))
#training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_Correct_L_training_velocity_1.pt',map_location=torch.device('cpu'))

#Fixed decay actions:
#decay_w = 10
decay_w = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_L_ExpTDecayParameter_3.pt').detach()
test_actions = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_L_ExpTDecay_actions_3.pt').detach()
training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_L_ExpTtraining_accuracy_3.pt',map_location=torch.device('cpu'))
training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Decay/Learn_Decay/Results/Supervised_L_ExpTtraining_velocity_3.pt',map_location=torch.device('cpu'))

#Basic actions
# test_actions = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Standard/Results/Supervised_Basic_final_actions_2.pt').detach()
# training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Standard/Results/Supervised_Basic_training_accuracy_2.pt',map_location=torch.device('cpu'))
# training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/Standard/Results/Supervised_Basic_training_velocity_2.pt',map_location=torch.device('cpu'))

dev = torch.device('cpu')

longer= True

if longer:
    n_RK_steps = 150
    tspan = [0, 0.6]
    n_zero_actions = 50
    zero_actions = torch.zeros(1, 2, n_zero_actions ).to(dev)
    test_actions = torch.cat([test_actions, zero_actions], dim=2)
else:
    n_RK_steps = 100 #150
    tspan = [0, 0.4] #[0, 0.6]

x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] #initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
#window_size = 20
#decay_w = 0.6


# Target endpoint, based on matlab - reach straight in front, atshoulder height
#x_hat = 0.792
#y_hat = 0
x_hat = 0 #0.392#0.792
y_hat = - 0.792 #- 0.2
t = np.linspace(tspan[0],tspan[1],n_RK_steps+1) # np.linspace(0,0.6,151)
training_steps = np.linspace(1,len(training_acc)-1,len(training_acc)-1)

# Use Exp time increase on exponential decay
test_arm = Spvsd_ExpDecay_Arm_model(tspan, x0, dev, decay_w, n_arms=1)

# Use non-learning decay arm model and simply set hyperparam to learnt parameter
#test_arm = Spvsd_Decay_Arm_model(tspan,x0,dev,decay_w,n_arms=1)

# Basic arm model:
#test_arm = Spvsd_Arm_model(tspan,x0,dev,n_arms=1)

dynamics = test_arm.perform_reaching(t_step,test_actions)
#dynamics = test_arm.perform_reaching(t_step,test_actions,99)


f_points = - dynamics.size()[0]


tor = torch.squeeze(dynamics[:,:,4:6])
angular_velocity = torch.squeeze(dynamics[:,:,2:4])

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
ax3.plot(t[:-1],test_actions[0,0,:])

ax3.set_title("u_1")

ax4 = fig1.add_subplot(324)

ax4.plot(t[:-1],test_actions[0,1,:])

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