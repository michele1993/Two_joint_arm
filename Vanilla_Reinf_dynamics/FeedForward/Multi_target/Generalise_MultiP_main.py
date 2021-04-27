import torch
from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import *
import numpy as np

actor_params = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_MultiReinf_Actor_2.pt")
critic_params = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_MultiReinf_critic_2.pt")
target_points = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_MultiReinf_TargetPoints_2.pt")

dev = torch.device('cpu')
n_RK_steps = 99
std = 0.01
n_arms = 1
max_u = 15000 # add it afterward to
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]
t_step = tspan[-1]/n_RK_steps
f_points = -1
DPG_ln_rate = 1000
general_eps = 1000
t_print = 10
n_targets = 3

agent = Reinf_Actor_NN(std,n_arms,1,dev,ln_rate = DPG_ln_rate) # add max_u afterward since Q trained with tanh output
critic = Critic_NN(n_arms,dev)

agent.load_state_dict(actor_params)
critic.load_state_dict(critic_params)

arm = Parall_Arm_model(tspan,x0,dev,n_arms=n_targets)

noise = torch.randn(1,2) * 1

target_states = target_points + noise

# Test DPG adapation:

tot_accuracy = []
tot_velocity = []

for ep in range(general_eps):

    tanh_actions = agent(target_points, True)

    actions = (tanh_actions * max_u).view(n_targets, 2, n_RK_steps).detach()

    _, thetas = arm.perform_reaching(t_step,actions)

    rwd = arm.multiP_compute_rwd(thetas,target_states[:,0],target_states[:,1], f_points).squeeze()

    sqrd_velocity = arm.compute_vel(thetas, f_points).squeeze()


    TargetQ = critic(target_states,tanh_actions,True) # one at the time or all in one update ?

    noise_a = torch.randn((3,198)) *0

    TargetQ_2 = critic(target_points, tanh_actions + noise_a, True)

    print(target_states)
    print(TargetQ)
    print(target_points)
    print(TargetQ_2)
    exit()


    agent.DPG_update(TargetQ)

    tot_accuracy.append(torch.sqrt(rwd))
    tot_velocity.append(torch.sqrt(sqrd_velocity))

    if ep % t_print == 0:

        print_acc = sum(tot_accuracy)/(t_print*3) # since use 3 target points
        print_vel = sum(tot_velocity)/(t_print*3)

        print("Ep: ",ep)
        print("Accuracy: ",print_acc)
        print("Velocity: ",print_vel,"\n")

        tot_accuracy = []
        tot_velocity = []


# USE THIS, to test trained Reinforce generalises a bit
# for target_state in target_points:
#
#     noise = torch.randn(1,2) * 0.01
#
#     target_state = target_state.view(1,2) + noise
#     tanh_actions = agent(target_state, True).view(1, 2, -1).detach()
#     actions = tanh_actions * max_u
#
#
#     _, thetas = arm.perform_reaching(t_step,actions)
#
#     rwd = arm.compute_rwd(thetas,target_state[0,0],target_state[0,1], f_points)
#     sqrd_velocity = arm.compute_vel(thetas, f_points)
#
#     accuracy = torch.sqrt(rwd)
#     velocity = torch.sqrt(sqrd_velocity)
#
#     print("Target: ", target_state)
#     print("Accuracy: ",accuracy)
#     print("Velocity: ",velocity,"\n")
