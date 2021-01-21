import torch
import numpy as np
from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Agent import Spvsd_FB_L_Agent
from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Arm_model import FB_L_Arm_model


# Regularising thor doesn't work (still get massive deceleration) as well as regularising acceleration only, though haven't played much with
# regulariser hyper-param, I guess could try regularising both (thor & acceleration), but maybe painful to tune hyper-params

episodes = 10000
ln_rate = 0.002 # may need a lower one, it's oscillating
n_RK_steps = 100
time_window = 10
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

arm = FB_L_Arm_model(tspan,x0,dev, n_arms=1)
agent = Spvsd_FB_L_Agent(dev,ln_rate= ln_rate)

ep_distance = []
ep_velocity = []

velocity_weight = 0.8
accel_weight = 0.0000005 #0.000005 getting closer
#thor_weight = 0.0001

training_accuracy= []
training_velocity = []

for ep in range(episodes):

    thetas, u_s, e_params = arm.perform_reaching(t_step,agent)

    # Compute squared distance for x and y coord, so that can optimise squared velocity
    x_sqrd_dist, y_sqrd_dist = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    sqrd_distance = torch.mean(x_sqrd_dist + y_sqrd_dist, dim=0, keepdim=True) # mean squared distance from target across window points for optimisation

    sqrd_dx, sqrd_dy = arm.compute_vel(thetas,0) # use 0 to select all time values from thetas, need all to compute norm of acceleration across entire reach

    # Compute velocity for points in window only, as needed for the optimisation
    window_srd_dx = sqrd_dx[f_points:,:]
    window_srd_dy = sqrd_dy[f_points:, :]

    sqrd_velocity = torch.mean( window_srd_dx + window_srd_dy,dim=0,keepdim=True)

    # Compute actual velocity
    entire_velocity = torch.sqrt(sqrd_dx + sqrd_dy)
    acceleration = arm.compute_accel(entire_velocity,t_step)

    # --------------- Need this, if wanna regularise thor -------------------------------------
    # tor = thetas[:, :, 4:6]
    # thr1_norm = torch.linalg.norm(tor[:,:,0])
    # thr2_norm = torch.linalg.norm(tor[:, :, 1])
    # --------------------------------------------------------------------------


    # sum of squared distance and weighted squared velocity
    loss = sqrd_distance + (sqrd_velocity * velocity_weight) + (torch.linalg.norm(acceleration) * accel_weight) # + (thr1_norm + thr2_norm) * thor_weight


    agent.update(loss)

    ep_distance.append(torch.mean(torch.sqrt(x_sqrd_dist + y_sqrd_dist)).detach()) # mean distance to assess performance
    # Take sqrt() of squared velocity to obtain actual velocity
    ep_velocity.append(torch.mean(torch.sqrt(window_srd_dx + window_srd_dy)).detach())



    if ep % t_print == 0:

        av_acc = (sum(ep_distance)/t_print)
        av_vel = (sum(ep_velocity) / t_print)

        training_accuracy.append(av_acc)
        training_velocity.append(av_vel)

        print("ep: ",ep)
        print("distance: ",av_acc)
        print("velocity: ",av_vel)
        print("acceleration norm: ", torch.norm(acceleration))
        ep_distance = []
        ep_velocity = []

        if av_acc <= 0.0005:
            break



torch.save(thetas, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_dynamics_2.pt')
torch.save(u_s, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_actions_2.pt')
torch.save(e_params, '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_expParams_2.pt')
torch.save(agent.state_dict(), '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_NNparameters_2.pt')
torch.save(training_accuracy,
           '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_accuracy_2.pt')
torch.save(training_velocity,
                      '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Results/Spvsd_FB_L_RegDecay_training_velocity_2.pt')