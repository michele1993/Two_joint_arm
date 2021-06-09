import torch
from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import *
import numpy as np

# After having trained model on multiple targets with REINFORCE, test DPG vs REINFORCE
# generalisation ability, doesn't work for 7 points

torch.manual_seed(1)

actor_params = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/Parallel_MultiReinf_Actor_2.pt",map_location=torch.device('cpu'))
critic_params = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/Parallel_MultiReinf_critic_2.pt",map_location=torch.device('cpu'))
target_points = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/Parallel_MultiReinf_TargetPoints_2.pt",map_location=torch.device('cpu'))

dev = torch.device('cpu')
n_RK_steps = 99
std = 0.01
n_arms = 1
max_u = 15000 # Q was learnt with full magnitude actions
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]
t_step = tspan[-1]/n_RK_steps
f_points = -1
DPG_ln_rate = 0.001
general_eps = 100#000
t_print = 10
n_targets = target_points.size()[0]


agent = Reinf_Actor_NN(std,n_arms, max_u,dev,ln_rate = DPG_ln_rate) # 1 is for max_u, add afterward since Q trained with tanh output
critic = Critic_NN(n_arms,dev)

agent.load_state_dict(actor_params)
critic.load_state_dict(critic_params)

arm = Parall_Arm_model(tspan,x0,dev,n_arms=n_targets)

noise = torch.randn(n_targets,2) * 1#.0001

target_states = target_points + noise

#target_states = torch.cat([target_states[0:1,:],target_states[2:,:]])


# Test DPG adapation:

tot_accuracy = []
tot_velocity = []

for ep in range(1,general_eps):

    Q_actions = agent(target_states, True)

    actions = Q_actions.view(n_targets, 2, n_RK_steps).detach()

    _, thetas = arm.perform_reaching(t_step,actions)

    rwd = arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points,n_arms).squeeze()


    sqrd_velocity = arm.compute_vel(thetas, f_points).squeeze()


    TargetQ = critic(target_states,Q_actions,True) # one at the time or all in one update ?

    print(target_states)
    print(TargetQ, "\n")

    TargetQ_2 = critic(target_points, Q_actions, True)

    print(target_points)
    print(TargetQ_2, "\n")
    exit()

    # Try to perturb actions to see if grad Q changes
    #noise_a = torch.randn((7,198)) *0
    #TargetQ_2 = critic(target_points, tanh_actions + noise_a, True)
    # print(target_states)
    # print(TargetQ)
    # print(target_points)
    # print(TargetQ_2)
    # exit()

    agent.DPG_update(TargetQ)

    tot_accuracy.append(torch.mean(torch.sqrt(rwd)))
    tot_velocity.append(torch.mean(torch.sqrt(sqrd_velocity)))

    if ep % t_print == 0:

        print_acc = sum(tot_accuracy)/(t_print)
        print_vel = sum(tot_velocity)/(t_print)

        print("Ep: ",ep)
        print("Accuracy: ",print_acc)
        print("Velocity: ",print_vel,"\n")

        tot_accuracy = []
        tot_velocity = []



# USE THIS, to test trained Reinforce generalises a bit
# arm = Parall_Arm_model(tspan,x0,dev,n_arms=1)
# for target_state in target_points:
#
#     noise = torch.randn(1,2) * 0.05
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
