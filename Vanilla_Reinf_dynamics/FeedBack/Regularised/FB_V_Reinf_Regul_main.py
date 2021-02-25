from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Agent import FB_Reinf_Agent
from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Parall_Arm_model import FB_Par_Arm_model
import torch
import numpy as np


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')


episodes = 15000
n_RK_steps = 100
t_print = 50
n_arms = 5000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -10


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

training_arm = FB_Par_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = FB_Reinf_Agent(dev=dev,n_arms= n_arms).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01
beta = 0.1 # 0.4

# Tried : --------------------------------------------------------------------------------------------
# Notice regularising euclidean accel not optimal as this is reflected at the end-point, rather than in controller

# accel_weight = 0.0000001; model learn, but still decelerates too much
#accel_weight = 0.000001 # model doesn't learn to stop, but deceleration is good

#One concern: when done successfully for supervised learning, I was clipping u to 5000, and it may have helped ?
#-----------------------------------------------------------------------------------------------------

#Now need to try:
accel_weight = 0.0000005

training_acc = []
training_vel = []

ep_rwd = []
ep_vel = []
train = True


for ep in range(episodes):


    thetas, u, decay_param = training_arm.perform_reaching(t_step,agent,train)

    sqrd_x, sqrd_y = training_arm.compute_distance(thetas,x_hat,y_hat, f_points)
    sqrd_distance = sqrd_x + sqrd_y
    sqrd_mean_distance = torch.mean(sqrd_distance,dim=0,keepdim=True)


    advantage = sqrd_mean_distance - avr_rwd
    avr_rwd += alpha * torch.mean(advantage)


    sqrd_v_dx,sqrd_v_dy = training_arm.compute_vel(thetas,0) # select all points across trajecotry, needed for norm of accelertion
    window_srd_dx = sqrd_v_dx[f_points:,:]
    window_srd_dy = sqrd_v_dy[f_points:, :]

    w_sqrd_velocity = window_srd_dx + window_srd_dy
    sqrd_mean_velocity = torch.mean(w_sqrd_velocity,dim=0,keepdim=True)

    entire_velocity = torch.sqrt(sqrd_v_dx + sqrd_v_dy)
    accelleration = training_arm.compute_accel(entire_velocity,t_step)
    norm_accel = torch.linalg.norm(accelleration,dim=0, keepdim=True) # compute norm of acceleration for each arm


    vel_adv = sqrd_mean_velocity - avr_vel
    avr_vel += alpha * torch.mean(vel_adv)

    weighted_adv = advantage + beta * vel_adv + accel_weight * norm_accel


    agent.update(weighted_adv)

    # Store accuracy and velocity over time window
    ep_rwd.append(torch.mean(torch.sqrt(sqrd_distance.detach())))
    ep_vel.append(torch.mean(torch.sqrt(w_sqrd_velocity.detach())))


    if ep % t_print == 0:

        av_acc = sum(ep_rwd)/t_print
        av_vel = sum(ep_vel)/t_print

        training_acc.append(av_acc)
        training_vel.append(av_vel)

        print("episode: ", ep)
        print("training distance: ",av_acc)
        print("max Accel: ", torch.max(accelleration))
        print("End velocity: ",  av_vel,'\n')
        ep_rwd = []
        ep_vel = []

        if av_acc <= 0.0010:
            break



        # if dev.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        #     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


torch.save(agent.state_dict(), '/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_parameters_Decay_Regul_3.pt')
torch.save(training_acc,'/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_TrainingAcc_Decay_Regul_3.pt')
torch.save(training_vel,'/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/Regularised/Results/FB_TrainingVel_Decay_Regul_3.pt')

train = False
tst_tspan = tspan #[0, 0.4 + (t_step* f_points)] # extend time of simulation to see if arm bounce back
test_arm = FB_Par_Arm_model(tst_tspan,x0,dev,n_arms=1)
agent.n_arms = 1
thetas, tst_u,decay_param = test_arm.perform_reaching(t_step, agent, train)

tst_sqrd_x, tst_sqrd_y = test_arm.compute_distance(thetas, x_hat, y_hat, f_points)
tst_sqrd_distance = tst_sqrd_x + tst_sqrd_y
test_rwd = torch.mean(torch.sqrt(tst_sqrd_distance))

tst_sqrd_dx, tst_sqrd_dy = test_arm.compute_vel(thetas,f_points)
tst_vel = torch.mean(torch.sqrt(tst_sqrd_dx + tst_sqrd_dy))
print("test accuracy: ",test_rwd)
print("test velocity: ", tst_vel)
