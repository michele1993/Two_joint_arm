from Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.Vanilla_Reinf_Agent import Reinf_Agent
import torch


episodes = 100000
n_parameters = 2
t_print = 10
n_arms = 50

# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0


arm = Parall_Arm_model(n_arms=n_arms,tspan = [0, 0.4])
agent = Reinf_Agent(n_parameters,n_arms)

ep_rwds = []
avr_rwd = 0
alpha =  0.01
t_step = 0.4/100


for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor

    t, thetas = arm.perform_reaching(t_step,actions)

    rwd = arm.compute_rwd(thetas,x_hat,y_hat)

    advantage = rwd - avr_rwd

    avr_rwd += alpha * (rwd - avr_rwd)

    agent.update(advantage)

    ep_rwds.append(rwd)

    if ep % t_print == 0:

        print(torch.mean(sum(ep_rwds)/t_print))
        ep_rwds = []

        #print(agent.mu_s)