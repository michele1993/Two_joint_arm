from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Agent import FB_Reinf_Agent
from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Parall_Arm_model import FB_Par_Arm_model
import torch

import numpy as np


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')


episodes = 100000
n_RK_steps = 100
t_print = 100
n_arms = 1#5000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = -15


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

training_arm = FB_Par_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = FB_Reinf_Agent(dev=dev,n_arms= n_arms).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01
beta = 0.4 # 0.1

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


    sqrd_v_dx,sqrd_v_dy = training_arm.compute_vel(thetas,f_points)
    sqrd_velocity = sqrd_v_dx + sqrd_v_dy
    sqrd_mean_velocity = torch.mean(sqrd_velocity,dim=0,keepdim=True)

    vel_adv = sqrd_mean_velocity - avr_vel
    avr_vel += alpha * torch.mean(vel_adv)

    weighted_adv = advantage + beta * vel_adv


    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(torch.sqrt(sqrd_distance.detach())))
    ep_vel.append(torch.mean(torch.sqrt(sqrd_velocity.detach())))


    if ep % t_print == 0:

        av_acc = sum(ep_rwd)/t_print
        av_vel = sum(ep_vel)/t_print

        training_acc.append(av_acc)
        training_vel.append(av_vel)

        print("episode: ", ep)
        print("training distance: ",av_acc)
        print("decay: ", torch.mean(decay_param[-1]))
        print("End velocity: ",  av_vel,'\n')
        ep_rwd = []
        ep_vel = []

        if av_acc <= 0.0002:
            break



        # if dev.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        #     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


torch.save(agent.state_dict(), '/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/FB_results/FB_parameters_Decay_NoAngAccelDecay_OneArm1.pt')
torch.save(training_acc,'/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/FB_results/FB_TrainingAcc_Decay_NoAngAccelDecay_OneArm1.pt')
torch.save(training_vel,'/home/px19783/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedBack/FB_results/FB_TrainingVel_Decay_NoAngAccelDecay_OneArm1.pt')

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


