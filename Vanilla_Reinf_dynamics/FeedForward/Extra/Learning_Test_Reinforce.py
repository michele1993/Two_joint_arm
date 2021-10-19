from Extra_models.Arm_model import Arm_model
from Vanilla_Reinf_dynamics.FeedForward.Extra.Individual_params.Vanilla_Reinf_Agent import Reinf_Agent
import numpy as np
import torch


# Test whether the Vanilla Reinforce algorithms is actually capable of learning, by asking to learn target control signal,
# (simpler learning problem) or whether it has a bug. It shows learning of target control signal. Then, test whether it
# can learn target angles

episodes = 100000
eval_points = 2
t_print = 500

# Target endpoint, based on matlab - reach strainght in front at shoulder height
x_hat = 0.792
y_hat = 0

arm = Arm_model(n_points = eval_points)
agent = Reinf_Agent(eval_points)

ep_rwds = []

#target_control = np.arange((eval_points - 1) *2).reshape(-1,2)

target_angles = np.arange((eval_points - 1) *2).reshape(-1,2)
target_angles[:,0] *= -1

avr_rwd= 0

alpha = 0.01

for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor

    t, thetas = arm.perfom_reaching(actions)

    y_ = thetas[1:,[0,1]]


    rwd = torch.Tensor(np.sqrt((y_ - target_angles)**2))


    #rwd = np.sqrt((actions - target_control)**2).detach()

    #dis_rwd = torch.tensor(rwd)
    #dis_rwd = agent.forward_dis_return(rwd)


    #old_mus = torch.clone(agent.mu_s)

    advantage = rwd - avr_rwd

    avr_rwd += alpha * (rwd - avr_rwd)

    agent.update(advantage)

    #print(sum(old_mus == agent.mu_s))

    ep_rwds.append(rwd)



    if ep % t_print == 0:

        print(sum(sum(ep_rwds)/t_print))
        ep_rwds = []

        #print(agent.mu_s[0,[0,1]])