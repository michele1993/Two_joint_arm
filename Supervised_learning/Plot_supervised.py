#from Supervised_learning.Supervised_Arm_Model import Spvsd_Arm_model
from Supervised_learning.Decay.Spvsd_Decay_Arm_Model import Spvsd_Decay_Arm_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# Supervised_Basic_2: 50000 eps, time window: 10 with no zero_actions, optimised accuracy & velocity,vel_weight = 0.8,ln_rate = 1 ,Viscosity * 15

# Supervised_decay_2 : 100000 eps, time window: 10 with no zero_actions, optimised accuracy & velocity plus decay in dynamical model (NOTE: decay was alos applied to angular velocity
# and to the control singal u, the latter not making much sense), decay_weight = 0.6, vel_weight = 0.8,ln_rate = 50 ,Viscosity * 15


test_actions = torch.load("/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Decay/Results/Supervised_Decay_actions_3.pt",map_location=torch.device('cpu')).detach()
#dynamics = torch.load("/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Decay/Results/Supervised_Decay_dynamics2.pt",map_location=torch.device('cpu')).detach()
training_acc = torch.load("/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Decay/Results/Supervised_Decay_training_accuracy_3.pt",map_location=torch.device('cpu'))
training_vel = torch.load("/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Decay/Results/Supervised_Decay_training_velocity_3.pt",map_location=torch.device('cpu'))



n_RK_steps = 100 #150
tspan = [0, 0.4] #[0, 0.6]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] #initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
dev = torch.device('cpu')
window_size = 10
decay_w = 0.6


# Target endpoint, based on matlab - reach straight in front, atshoulder height
x_hat = 0.792
y_hat = 0
t = np.linspace(tspan[0],tspan[1],n_RK_steps+1) # np.linspace(0,0.6,151)
training_steps = np.linspace(1,len(training_acc)-1,len(training_acc)-1)




test_arm = Spvsd_Decay_Arm_model(tspan,x0,dev,decay_w,n_arms=1)
#zero_actions = torch.zeros(1, 2, 50).to(dev)
#test_actions = torch.cat([test_actions, zero_actions], dim=2)


_, dynamics = test_arm.perform_reaching(t_step,test_actions,99)


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
#
# ax6 = fig1.add_subplot(323)
#
# ax6.plot(t,tst_velocity[:,0])
#
# ax6.set_title("dt_1")
#
# ax7 = fig1.add_subplot(324)
#
# ax7.plot(t,tst_velocity[:,1])
#
# ax7.set_title("dt_2")



plt.show()

