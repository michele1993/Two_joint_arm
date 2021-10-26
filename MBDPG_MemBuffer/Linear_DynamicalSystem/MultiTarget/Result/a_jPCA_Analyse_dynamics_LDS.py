import jPCA
from jPCA.util import plot_projections
import torch
import numpy as np
from MBDPG_MemBuffer.Linear_DynamicalSystem.MultiTarget.MultiT_MBDPG_LDS_agent import Linear_DS_agent
from MBDPG_MemBuffer.Linear_DynamicalSystem.MultiTarget.MultiT_MBDPG_complexLDS_agent import Complex_Linear_DS_agent
from Supervised_learning.Feed_Forward.LDS_analysis.SPVD_LDS_agent import Linear_DS_agent
#from Supervised_learning.Feed_Forward.LDS_MultiTarget.complex_SPVD_LDS_agent import SPVSD_Complex_Linear_DS_agent
import matplotlib.pyplot as plt

jpca = jPCA.JPCA(num_jpcs=2)

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

    accuracy = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_training_acc_test_s913_3_oneArmOptim.pt")
    agent_NN = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_test_s913_3_oneArmOptim.pt")
    agent_D = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_D_test_s913_3_oneArmOptim.pt")
    agent_P = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/MultiTarget/Result/MT_LDS_MBDPG_actor_P_test_s913_3_oneArmOptim.pt")

    agent = Linear_DS_agent(t_step,n_parametrised_steps,0,0,n_targets,dev).to(dev)
    agent.D = agent_D
    agent.P = agent_P

elif learning_type == "Supervised":

    accuracy = None
    agent_NN = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Forward/LDS_analysis/Result/SPVSD_LDS_parameters_1")
    agent = Linear_DS_agent(t_step,n_parametrised_steps,0,n_targets,dev)

agent.load_state_dict(agent_NN)


target_states = torch.load('/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))
rand_indx = torch.randint(n_targets,(1,))
actions, h_state = agent(target_states)

time_rng = (torch.linspace(0,tspan[1] - t_step,n_parametrised_steps) *1000).type(torch.int).tolist()

rand_target_dynamics = h_state.squeeze().detach()#[:,:,rand_indx].squeeze().detach()


dynamics = []

for i in range(0,10):

    dynamics.append(rand_target_dynamics[:,:,i].numpy())



(projected,
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(dynamics, times=time_rng, tstart=0, tend=395)


# Plot the projected data
# x_idx and y_idx control which columns of the data are shown.
# For example, to plot the second jPCA plane, use x_idx=2, y_idx=3
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_projections(projected, axis=axes[0], x_idx=0, y_idx=1)
#plot_projections(projected, axis=axes[1], x_idx=2, y_idx=3)

axes[0].set_title("jPCA Plane 1")
axes[1].set_title("jPCA Plane 2")
plt.tight_layout()
plt.show()

