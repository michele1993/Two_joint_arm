#from Supervised_learning.Feed_Forward.Supervised_Arm_Model import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.LDS_MultiTarget.SPVSD_stopping_LDS_FF_parall_arm import Spvsd_Arm_model
from Supervised_learning.Feed_Forward.LDS_analysis.SPVD_LDS_agent import Spvsd_Linear_DS_agent
from Supervised_learning.Feed_Forward.LDS_MultiTarget.complex_SPVD_LDS_agent import SPVSD_Complex_Linear_DS_agent
import numpy as np
import torch

torch.manual_seed(913)  # FIX SEED

dynam_decay = 0.5


allow_complex = False

#Perform supervised learning using the first attempted approach, namely, using the dynamical model as provided by Berret et al. and using the
# distance and velocity as cost function, with no regularisation or decay, where the agent is a NN mapping desired states to actions.

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')
episodes = 250000
n_RK_steps = 99
time_window = 29
n_parametrised_steps = n_RK_steps #-time_window
t_print = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps
f_points = -time_window -1
velocity_weight = 0.2#0.005
max_u = 15000 # 15000
th_error = 0.01#0.025 # i.e. same accuracy as DPG at test
n_targets = 50
n_arms = 1

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


arm = Spvsd_Arm_model(tspan,x0,dynam_decay,dev, n_arms=n_targets)

#target_states = torch.load('/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))
target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))

actor_ln = 0.0025

if allow_complex:
    agent = SPVSD_Complex_Linear_DS_agent(t_step, n_parametrised_steps, actor_ln, n_targets, dev).to(dev)
    #actor_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1_complex'
    actor_file = '/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1_complex'

else:
    agent = Spvsd_Linear_DS_agent(t_step,n_parametrised_steps,actor_ln,n_targets,dev).to(dev)
    #actor_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1'
    actor_file = '/home/px19783/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1'


agent.apply(agent.small_weight_init)


ep_distance = []
ep_velocity = []


training_accuracy= []
training_velocity = []


for ep in range(1,episodes):


    actions,_ = agent(target_states)
    actions = actions * max_u


    thetas = arm.perform_reaching(t_step,actions)


    # NOT SURE GOOD IDEA, maybe better to optim sqrt (i.e. actual distance):
    # Compute squared distance for x and y coord, so that can optimise that and then apply sqrt() to obtain actual distance as a measure of performance
    rwd = arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)

    velocity = arm.compute_vel(thetas,f_points)

    loss = torch.mean(rwd,dim=0,keepdim=True) + (torch.mean(velocity,dim=0,keepdim=True) * velocity_weight)

    agent.update(loss)

    ep_distance.append(torch.mean(torch.sqrt(rwd).detach(),dim=1)) # mean distance to assess performance
    ep_velocity.append(torch.mean(torch.sqrt(velocity).detach()))


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


torch.save(agent.state_dict(),actor_file)