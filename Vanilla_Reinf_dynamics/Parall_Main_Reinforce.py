from Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.Vanilla_Reinf_Agent import Reinf_Agent
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')


episodes = 15000
n_RK_steps = 100
time_window_steps = 20
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 50
n_arms = 5000
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps #[- time_window_steps, -1] # number of final points to average across for distance to target and velocity
vel_weight = 0.8


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

training_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = Reinf_Agent(n_parametrised_steps,dev,n_arms= n_arms).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01

ep_rwd = []
ep_vel = []


for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor
    zero_actions = torch.zeros(n_arms, 2, time_window_steps).to(dev)
    actions = torch.cat([actions, zero_actions], dim=2)

    t, thetas = training_arm.perform_reaching(t_step,actions)

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    advantage = rwd - avr_rwd
    avr_rwd += alpha * torch.mean(advantage)

    velocity_adv = velocity - avr_vel
    avr_vel += alpha * torch.mean(velocity_adv)

    weighted_adv = advantage + vel_weight * velocity_adv #(velocity_adv + tau_adv)

    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))


    if ep % t_print == 0:

        print("episode: ", ep)
        print("training accuracy: ",sum(ep_rwd)/t_print)
        print("training velocity: ", sum(ep_vel)/t_print)
        ep_rwd = []
        ep_vel = []



        # if dev.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        #     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')



tst_tspan =tspan #[0, 0.4 + (t_step* f_points)] # extend time of simulation to see if arm bounce back
test_arm = Parall_Arm_model(tst_tspan,x0,dev,n_arms=1)

test_actions = torch.unsqueeze(agent.test_actions(),0).detach()
zero_actions = torch.zeros(1,2,time_window_steps).to(dev)
test_actions = torch.cat([test_actions,zero_actions],dim=2)

torch.save(test_actions, 'Results/test_actions1_av_vel_20points.pt')

agent.gaussian_convol(test_actions)
# add some zero input for extra time


t_t, t_y = test_arm.perform_reaching(t_step,test_actions)


torch.save(t_y, 'Results/test_dynamics1_av_vel_20pointsl.pt')


tst_accuracy = test_arm.compute_rwd(t_y,x_hat,y_hat,f_points)
tst_velocity = test_arm.compute_vel(t_y, f_points)



print("Test accuracy: ",tst_accuracy)
print("Test velocity", tst_velocity)


