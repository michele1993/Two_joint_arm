import torch
import numpy as np
from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from MBDPG_MemBuffer.Linear_DynamicalSystem.MultiTarget.stopping_LDS_FF_parall_arm import Stopping_FF_Parall_Arm_model
from MBDPG_MemBuffer.Linear_DynamicalSystem.MultiTarget.MultiT_MBDPG_LDS_agent import Linear_DS_agent
from MBDPG_MemBuffer.Linear_DynamicalSystem.MultiTarget.MultiT_MBDPG_complexLDS_agent import Complex_Linear_DS_agent
from Supervised_learning.Feed_Forward.LDS_analysis.SPVD_LDS_agent import Spvsd_Linear_DS_agent
import matplotlib.pyplot as plt
#from Supervised_learning.Feed_Forward.LDS_analysis.SPVD_LDS_agent import Linear_DS_agent
#from Supervised_learning.Feed_Forward.LDS_MultiTarget.complex_SPVD_LDS_agent import SPVSD_Complex_Linear_DS_agent

font = {'family' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

learning_types = ["Real", "Complex", "Supervised"]
learning_type = learning_types[2]

print(learning_type)

dev = torch.device('cpu')

n_parametrised_steps = 99
tspan = [0, 0.4]
t_step = tspan[-1] / n_parametrised_steps
n_targets = 50
n_hidden_u = 10
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]
max_u = 15000
est_y_size = 4

if learning_type == "Complex":

    accuracy = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_training_acc_test_s913_1_oneArmOptim_complex.pt")
    agent_NN = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_test_s913_1_oneArmOptim_complex.pt")
    agent_A = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_A_test_s913_1_oneArmOptim_complex.pt")
    agent = Complex_Linear_DS_agent(t_step,n_parametrised_steps,0,0,n_targets,dev).to(dev)
    agent.A = agent_A

elif learning_type == "Real":

    accuracy = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_training_acc_test_s913_1_oneArmOptim_stop.pt")
    agent_NN = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_test_s913_1_oneArmOptim_stop.pt")
    agent_D = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_D_test_s913_1_oneArmOptim_stop.pt")
    agent_P = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_P_test_s913_1_oneArmOptim_stop.pt")

    agent = Linear_DS_agent(t_step,n_parametrised_steps,0,0,n_targets,dev).to(dev)
    agent.D = agent_D
    agent.P = agent_P

elif learning_type == "Supervised":

    accuracy = None
    agent_NN = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1")
    agent = Spvsd_Linear_DS_agent(t_step,n_parametrised_steps,0,n_targets,dev)


agent.load_state_dict(agent_NN)


target_states = torch.load('/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))

#rand_indx = torch.randint(n_targets,(1,))
rand_indx = 0

actions, h_state = agent(target_states)


time_rng = torch.linspace(0,tspan[1] - t_step,n_parametrised_steps) # don't plot hidden dynamics at t = 0.4



rand_target_dynamics = h_state[:,:,rand_indx].squeeze().detach()
rand_target_actions = actions[rand_indx,:,:].squeeze().detach() * max_u


fig, axs = plt.subplots(2,5)
fig.set_size_inches(15, 9)


for i in range(2):

    for e in range(5):

        axs[i,e].plot(time_rng,rand_target_dynamics[:,5*i+e])
        axs[i,e].spines['right'].set_visible(False)
        axs[i,e].spines['top'].set_visible(False)

        if i < 1:
            axs[i,e].set_xticks([])

        if i > 0:
            axs[i,e].set_xlabel("time")


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.95,
                    top=0.95,
                    wspace=0.6,
                    hspace=0.2)

fig2 = plt.figure()
font = {'family' : 'normal',
        'size'   : 8}

plt.rc('font', **font)
plt.plot(time_rng,rand_target_actions.T)

# Increase signal through a Gaussian matrix

# G_matrix = torch.randn((100,n_hidden_u))
#
#
# rand_projection = G_matrix @ rand_target_dynamics.T
#
# fig2, axs2 = plt.subplots(10,10)
#
# fig2.set_size_inches(15, 9)



# for i in range(10):
#
#     for e in range(10):
#
#         axs2[i,e].plot(time_rng,rand_projection[i,:])
#         axs2[i,e].spines['right'].set_visible(False)
#         axs2[i,e].spines['top'].set_visible(False)
#
#
# plt.show()


# Initialise arm model and test accuracy

#arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_targets)
arm = Stopping_FF_Parall_Arm_model(tspan, x0, 0.5,dev, n_arms=n_targets)


simulator_actions = actions * max_u

_, thetas = arm.perform_reaching(t_step, simulator_actions.detach())


thetas_vel = thetas[:,rand_indx:rand_indx+1,0:est_y_size]

# extract two angles and two angle vel for all arms as target to the estimated model
thetas = thetas[-1:,:,0:est_y_size]

rwd = torch.sqrt(arm.multiP_compute_rwd(thetas, target_states[:, 0:1], target_states[:, 1:2], -1, 1)).squeeze()

vel = arm.compute_vel(thetas_vel, -99)
accel = arm.compute_accel(vel,t_step)

print(rwd[rand_indx])

fig_3 = plt.figure()
plt.plot(time_rng,vel.squeeze())

fig_4 = plt.figure()
plt.plot(time_rng[:-1],accel.squeeze())



plt.show()



