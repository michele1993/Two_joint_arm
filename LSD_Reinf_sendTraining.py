from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.Linear_DynamicalSystem.SingleTarget.Linear_DS_ReinfAgent import Linear_DS_agent
import torch
import numpy as np
import argparse

# trial inputs: -s 0 -d 0.0113 -a 5e-04 -i 1 -hs False
# or -s 37 -d 0.017 -a 5e-04 -i 1 -hs False

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?')
# Default values represent best values from hyperparam search:
parser.add_argument('--actorLr',   '-a', type=float, nargs='?')
parser.add_argument('--std',   '-d', type=float, nargs='?')
parser.add_argument('--counter',   '-i', type=int, nargs='?')
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)


args = parser.parse_args()
seed = args.seed
i = args.counter
actor_ln = args.actorLr
std = args.std
search_hyperParam = args.hyperparam_search

torch.manual_seed(seed)
dev = torch.device('cpu')

if not search_hyperParam:

    accuracy_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Linear_DynamicalSystem/SingleTarget/Result/LDS_Reinf_training_acc_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim.pt'
    actor_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Linear_DynamicalSystem/SingleTarget/Result/LDS_Reinf_actor_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim.pt'
    episodes = 10001
else:

    accuracy_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Linear_DynamicalSystem/SingleTarget/HyperParam_tuning/Result/LDS_Reinf_training_acc_hyperTuning_s' + str(
        seed) + '_' + str(i) + '_oneArm_2.pt'
    episodes = 35000#7501 #5001

n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 10#0 #50#10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps -1 # use last point with no zero action
vel_weight = 0.005
max_u = 15000 #10000
std_decay = 0.99


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0
target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


training_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = Linear_DS_agent(std,max_u,t_step,n_parametrised_steps,actor_ln,n_arms,dev).to(dev)
agent.apply(agent.small_weight_init)



best_acc = 50

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []



for ep in range(1,episodes):


    actions = agent(target_state, test = False) # may need converting to numpy since it's a tensor
    t, thetas = training_arm.perform_reaching(t_step,actions)

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % 100 == 0: # ep % 10 == 0: # decays works better if applied every 10 eps
        std *= std_decay


    if ep % t_print == 0:

        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ",print_acc)
        print("training velocity: ", print_vel,"\n")

        training_acc.append(print_acc)
        training_vel.append(print_vel)


        ep_rwd = []
        ep_vel = []

if not search_hyperParam:
    torch.save(training_acc, accuracy_file)
    #torch.save(agent.state_dict(), actor_file)

else:
    training_acc.append(std)
    training_acc.append(actor_ln)
    torch.save(training_acc, accuracy_file)