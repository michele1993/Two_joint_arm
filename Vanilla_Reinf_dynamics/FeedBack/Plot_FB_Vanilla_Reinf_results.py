from safety_checks.Video_arm_config import Video_arm
from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Parall_Arm_model import FB_Par_Arm_model
from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Agent import FB_Reinf_Agent


import torch
import numpy as np
import matplotlib.pyplot as plt


# Time decay learnt by the NN with or without regularisation on accel

# ----------------------------------------------------------------------------------------------------
# Decay_Regul_1: used accel_weight = 0.0000001; model learn, but still decelerates too much
# Decay_Regul_2: used accel_weight = 0.000001 # model doesn't learn to stop, but deceleration is good
#-------------------------------------------------------------------------------------------------------

training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_TrainingAcc_Decay_Regul_3.pt')
training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_TrainingVel_Decay_Regul_3.pt')
parameters = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_parameters_Decay_Regul_3.pt')


dev = torch.device('cpu')
number_arms = 1

agent = FB_Reinf_Agent(dev,number_arms)
agent.load_state_dict(parameters)

longer= True

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

# Use linear time increase on exponential decay
test_arm = FB_Par_Arm_model(tspan,x0,dev,n_arms=number_arms)

dynamics, test_actions,_ = test_arm.perform_reaching(t_step,agent, False) # train as False
dynamics = dynamics.detach()
test_actions = test_actions.detach()


f_points = - dynamics.size()[0]



# Compute final accuracy
dist_x, dist_y = test_arm.compute_distance(dynamics,x_hat,y_hat,f_points)
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