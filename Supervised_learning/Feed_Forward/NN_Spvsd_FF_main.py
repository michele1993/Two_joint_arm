#from Supervised_learning.Feed_Forward.Supervised_Arm_Model import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.LDS_MultiTarget.SPVSD_stopping_LDS_FF_parall_arm import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.NN_Spvsd_FF_agent import Actor_NN
from Supervised_learning.Feed_Forward.LDS_analysis.SingleT_SPVD_LDS_agent import SingTSpvsd_Linear_DS_agent
import numpy as np
import torch

torch.manual_seed(913)  # FIX SEED

dynam_decay = 0.5
#Perform supervised learning using the first attempted approach, namely, using the dynamical model as provided by Berret et al. and using the
# distance and velocity as cost function, with no regularisation or decay, where the agent is a NN mapping desired states to actions.

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')
episodes = 25000
n_RK_steps = 99
time_window = 29
n_parametrised_steps = n_RK_steps #-time_window
t_print = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window -1
ln_rate_a = 0.005
velocity_weight = 0#0.5#0.005# 0.2 not working
max_u = 15000 # 15000
th_error = 0.01#0.025 # i.e. same accuracy as DPG at test


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


#arm = Spvsd_Arm_model(tspan,x0,dev, n_arms=1)
arm = Spvsd_Arm_model(tspan,x0,dynam_decay,dev, n_arms=1)

test_LDS = True

# Initialise actor
if test_LDS:

    #actor_ln = 0.001#0.0002525000018067658
    actor_ln = 0.0005
    agent = SingTSpvsd_Linear_DS_agent(t_step,n_parametrised_steps,actor_ln,dev).to(dev)
    agent.apply(agent.small_weight_init)
    actor_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_SingleTarget'


else:
    agent = Actor_NN(max_u,dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
    agent.apply(agent.small_weight_init)



ep_distance = []
ep_velocity = []


training_accuracy= []
training_velocity = []

for ep in range(1,episodes):

    if test_LDS:
        actions = agent(target_state)[0].view(1,2,-1)
        actions = actions * max_u
    else:
        actions = agent(target_state).view(1, 2, -1)

    thetas = arm.perform_reaching(t_step,actions)

    # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
    # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
    rwd = arm.compute_rwd(thetas, x_hat,y_hat, f_points)
    velocity = arm.compute_vel(thetas,f_points)

    loss = torch.mean(rwd) + torch.mean(velocity * velocity_weight)

    agent.update(loss)

    ep_distance.append(torch.sqrt(rwd).detach()) # mean distance to assess performance
    ep_velocity.append(torch.mean(torch.sqrt(velocity)).detach())


    if ep % t_print == 0:

        av_acc = (sum(ep_distance)/t_print)
        av_vel = (sum(ep_velocity) / t_print)

        training_accuracy.append(av_acc)
        training_velocity.append(av_vel)

        print("ep: ",ep)
        print("distance: ",av_acc)
        print("velocity: ",av_vel)

        # if av_acc <= th_error:
        #     break

        ep_distance = []
        ep_velocity = []


if test_LDS:

    torch.save(agent.state_dict(),actor_file)

#else:
    # torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/Results/NN_Spvsd_FF_agent_s1_BestParams.py')
    # torch.save(training_accuracy,'/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/Results/NN_Spvsd_FF_training_accuracy_s1_BestParams.py')
    # torch.save(training_velocity,'/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/Results/NN_Spvsd_FF_training_velocity_s1_BestParams.py')
