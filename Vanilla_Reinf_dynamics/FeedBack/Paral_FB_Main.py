from Vanilla_Reinf_dynamics.FeedBack.FB_V_Reinf_Agent import FB_Reinf_Agent
from Vanilla_Reinf_dynamics.FeedBack.FB_Parall_Arm_model import FB_Par_Arm_model
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
f_points = -20


# Target endpoint, based on matlab - reach straight in front, at shoulder height
t1_hat = 0
t2_hat = 0

x_hat = 0.792
y_hat = 0

training_arm = FB_Par_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = FB_Reinf_Agent(dev=dev).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01
beta = 0.8

ep_rwd = []
ep_vel = []
ep_dist =[]
train = True


for ep in range(episodes):


    thetas, u = training_arm.perform_reaching(t_step,agent,train)

    #rwd = training_arm.compute_rwd(thetas,t1_hat,t2_hat)
    rwd = training_arm.compute_distance(thetas,x_hat,y_hat, f_points)
    #rwd = agent.forward_dis_return(rwd,n_RK_steps)
    #rwd = agent.compute_discounted_returns(rwd,f_points)

    advantage = rwd - avr_rwd
    avr_rwd += alpha * torch.mean(advantage)


    velocity = training_arm.compute_vel(thetas,f_points)
    vel_adv = velocity - avr_vel
    avr_vel += alpha * torch.mean(vel_adv)


    weighted_adv = advantage #+ beta * vel_adv


    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(rwd))
    #ep_dist.append(torch.mean(distance))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))


    if ep % t_print == 0:

        print("episode: ", ep)
        #print("training accuracy: ",sum(ep_dist)/t_print)
        print("training loss: ",sum(ep_rwd)/t_print)
        print("effectors: ", torch.mean(u))
        print("training velocity: ", sum(ep_vel)/t_print)
        ep_rwd = []
        ep_dist =[]
        ep_vel = []



        # if dev.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        #     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')



train = False
tst_tspan =tspan #[0, 0.4 + (t_step* f_points)] # extend time of simulation to see if arm bounce back
test_arm = FB_Par_Arm_model(tst_tspan,x0,dev,n_arms=1)
thetas, tst_u = test_arm.perform_reaching(t_step, agent, train)
rwd = training_arm.compute_rwd(thetas, t1_hat, t2_hat)

print(rwd[-1])

torch.save(agent.state_dict(), 'FB_results/FB_parameters_1.pt')
