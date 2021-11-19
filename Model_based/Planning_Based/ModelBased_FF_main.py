from Model_based.MB_Arm_model import MB_FF_Arm_model
from Model_based.MB_NN_Agent import MB_Actor_NN
from Model_based.Model_based_alg import MB_alg
import torch
import numpy as np

torch.manual_seed(0)
dev = torch.device('cpu')

Overall_episodes = 500
Model_episodes = 100
n_RK_steps = 99
time_window = 0
n_parametrised_steps = n_RK_steps -time_window
t_print = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window -1
ln_rate_a = 0.01 #0.01 works best #works well: 0.001 # 0.00001
velocity_weight = 0.005
max_u = 15000
th_error = 0.01#0.025
n_arms = 1#10 #100
Model_ln_rate = 0.005 #0.1 works best  #works well0.05#0.01 #0.08
std = 0.01

# Issues: if I use a large ln_rate it converges very quickly but with wrong parameters

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


target_arm = MB_FF_Arm_model(False,tspan,x0,dev, n_arms=n_arms)
estimated_arm = MB_FF_Arm_model(True,tspan,x0,dev, n_arms=n_arms,ln_rate = Model_ln_rate)

agent = MB_Actor_NN(max_u,dev,Output_size= n_parametrised_steps*2, ln_rate= ln_rate_a)
agent.apply(agent.small_weight_init)

MB_alg = MB_alg(estimated_arm,agent ,t_step, n_parametrised_steps,velocity_weight, th_error)

ep_distance = []
ep_model_ups = []
ep_actor_ups = []

for ep in range(1,Overall_episodes):


    actions = agent(target_state).view(1,2,n_parametrised_steps).detach()

    # exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)
    # actions = (det_actions + exploration)


    target_ths = target_arm.perform_reaching(t_step, actions)

    rwd = target_arm.compute_rwd(target_ths, target_state[0,0], target_state[0,1], -1)
    velocity = target_arm.compute_vel(target_ths, -1)

    acc = torch.sqrt(rwd)
    ep_distance.append(acc)

    # if torch.mean(torch.sqrt(rwd)) > th_error:

    print("Rollouts: ", ep)
    print("Overall Accuracy: ", acc)
    print("Overall Velocity: ", torch.mean(torch.sqrt(velocity)), "\n")



    ep_modelUp = MB_alg.update_model(actions.detach(), target_ths, target_arm)
    ep_model_ups.append(ep_modelUp)
    print("Eps to update model: ", ep_modelUp)

    ep_actorUp = MB_alg.update_actor(target_state)
    ep_actor_ups.append(ep_actorUp)



        # print(target_arm.beta)
        # print(estimated_arm.beta,"\n")
        # print(target_arm.omega)
        # print(estimated_arm.omega)
        # print(target_arm.alpha)
        # print(target_arm.omega)
        #print(target_arm.F)











