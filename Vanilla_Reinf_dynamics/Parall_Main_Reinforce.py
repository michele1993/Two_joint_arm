from Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.Vanilla_Reinf_Agent import Reinf_Agent
import torch
from safety_checks.Video_arm_config import Video_arm
import numpy as np


episodes = 1000
n_RK_steps = 100
n_parametrised_steps = n_RK_steps
t_print = 50
n_arms = 5000
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dev = torch.device('cpu')

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

tspan = [0, 0.4]


training_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = Reinf_Agent(n_parametrised_steps,dev,n_arms= n_arms).to(dev)

ep_rwds = []
avr_rwd = 0
alpha =  0.01
t_step = tspan[-1]/n_RK_steps



for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor

    t, thetas = training_arm.perform_reaching(t_step,actions)

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat)

    advantage = rwd - avr_rwd

    avr_rwd += alpha * (rwd - avr_rwd)

    agent.update(advantage)

    ep_rwds.append(rwd)

    if ep % t_print == 0:

        print("episode: ", ep)
        print("training accuracy: ",torch.mean(sum(ep_rwds)/t_print),'\n')
        ep_rwds = []

        if dev.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


test_arm = Parall_Arm_model(tspan,x0,dev,n_arms=1)
test_actions = torch.unsqueeze(agent.test_actions(),0).detach()

t_t, t_y = test_arm.perform_reaching(t_step,test_actions)

tst_accuracy = test_arm.compute_rwd(t_y,x_hat,y_hat)

print("Test accuracy: ",tst_accuracy)


#video1 = Video_arm(test_arm, np.squeeze(t_y.numpy()), np.array(t_t),fps = 60)
#video1.make_video()

        #print(agent.mu_s)