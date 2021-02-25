import torch
import numpy as np
from Supervised_learning.Feed_Back.FB_L_Decay.Regularised.Parallel_HyperParams.Spvsd_Parallel_Agent import Spvsd_FB_Parallel_Agent
from Supervised_learning.Feed_Back.FB_L_Decay.Spvsd_FB_L_Decay_Arm_model import FB_L_Arm_model

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dev = torch.device('cpu')

episodes = 10000
ln_rate = 0.002 # may need a lower one, it's oscillating
n_RK_steps = 100
time_window = 10
t_print = 10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window

# initialise a grid of hyper-parameters
n_hyparm = 1000
hyParam_range1 = torch.linspace(0,0.001,n_hyparm).to(dev)

#hyParam_range2 = torch.linspace(0,0.00005,n_hyparm).to(dev)

n_models = n_hyparm #* n_hyparm

#x_grid,y_grid = torch.meshgrid(hyParam_range1,hyParam_range2)# note y_grid is the transpose of x_grid

# Make size consistent with output of dynamical model to allow element-wise multiplication
hyparm_w1 = hyParam_range1.reshape(1,-1,1) #x_grid.reshape(1,-1,1)
# hyparm_w2 = y_grid.reshape(1,-1,1)


# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0


arm = FB_L_Arm_model(tspan,x0,dev, n_arms=n_models)
agent = Spvsd_FB_Parallel_Agent(n_models,dev,ln_rate= ln_rate).to(dev)

ep_distance = []
ep_velocity = []

velocity_weight = 0.4 #0.8

training_accuracy= []
training_velocity = []

for ep in range(episodes):

    thetas, u_s, e_params = arm.perform_reaching(t_step,agent)


    # Compute squared distance for x and y coord, so that can optimise squared velocity
    x_sqrd_dist, y_sqrd_dist = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    sqrd_distance = torch.mean(x_sqrd_dist + y_sqrd_dist, dim=0, keepdim=True) # mean squared distance from target across window points for optimisation

    window_srd_dx, window_srd_dy = arm.compute_vel(thetas,f_points) # use 0 to select all time values from thetas, need all to compute norm of acceleration across entire reach
    sqrd_velocity = torch.mean( window_srd_dx + window_srd_dy,dim=0,keepdim=True)

    # --------------- Compute the norm of thor and control signal, u -------------------------------------
    tor = thetas[:, :, 4:6]
    thr1_norm = torch.linalg.norm(tor[:,:,0], dim=0, keepdim=True)
    thr2_norm = torch.linalg.norm(tor[:, :, 1], dim=0, keepdim=True)

    # d_thor = thetas[:,:,6:8]
    # d_thr1_norm = torch.linalg.norm(d_thor[:,:,0],dim=0, keepdim=True)
    # d_thr2_norm = torch.linalg.norm(d_thor[:, :, 1],dim=0, keepdim=True)

    # u1_norm = torch.linalg.norm(u_s[:,:,0],dim=0, keepdim=True)
    # u2_norm = torch.linalg.norm(u_s[:, :, 1],dim=0, keepdim=True)

    # --------------------------------------------------------------------------


    # sum of squared distance and weighted squared velocity
    loss = sqrd_distance + (sqrd_velocity * velocity_weight) + (thr1_norm + thr2_norm) * hyparm_w1 #+ (d_thr1_norm + d_thr2_norm) #* hyparm_w2 #(u1_norm + u2_norm) * hyparm_w2


    #  sum all in order to feed to optimiser, this is not a problem since each model is independent parameterwise, so not risk optim of one affecting the other
    agent.update(torch.sum(loss))

    ep_distance.append(torch.mean(torch.sqrt(x_sqrd_dist + y_sqrd_dist),dim=0).detach()) # mean distance to assess performance
    # Take sqrt() of squared velocity to obtain actual velocity
    ep_velocity.append(torch.mean(torch.sqrt(window_srd_dx + window_srd_dy),dim=0).detach())


    if ep % t_print == 0 and ep >0:

        block_acc = torch.mean(torch.cat(ep_distance,dim=1),dim=1)
        block_vel = torch.mean(torch.cat(ep_velocity,dim=1),dim=1)

        training_accuracy.append(block_acc)
        training_velocity.append(block_vel)

        #best_acc , b_idx = torch.min(block_acc, dim=0)

        #rnd_idx = torch.randint(n_hyparm * n_hyparm, (1,))
        rnd_idx = torch.randint(n_hyparm,(1,))

        print("ep: ",ep)
        print("random model distance: ", block_acc[rnd_idx])
        print("random model velocity: ",block_vel[rnd_idx])
        print("equivalent hyper_parm1: ",hyparm_w1[:,rnd_idx])
        #print("equivalent hyper_parm2: ", hyparm_w2[:, rnd_idx])
        ep_distance = []
        ep_velocity = []

final_srd_dx, final_srd_dy = arm.compute_vel(thetas,0) # use 0 to select all time values from thetas, need all to compute norm of acceleration across entire reach
sqrd_velocity = torch.sqrt( final_srd_dx + final_srd_dy)

final_accell = arm.compute_accel(sqrd_velocity,t_step).detach()



torch.save(training_accuracy,
           '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_training_accuracy_4.pt')
torch.save(training_velocity,
                      '/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_training_velocity_4.pt')

torch.save(hyparm_w1,'/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_hyparamW1_4.pt')
#torch.save(hyparm_w2,'/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_hyparamW2_4.pt')

torch.save(final_accell,'/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_final_accelleration_4.pt')