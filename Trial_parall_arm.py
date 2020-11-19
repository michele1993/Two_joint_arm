from Parall_Arm_model import Parall_Arm_model
from Arm_model import Arm_model
import numpy as np
import torch

n_arms = 2
tspan = [0, 0.4]
paralell_arms1 = Parall_Arm_model(n_arms=n_arms, tspan = tspan)

n_arms2 = 3
paralell_arms2 = Parall_Arm_model(n_arms= n_arms2, tspan = tspan)

#arm1 = Arm_model(tspan = tspan)




#



# ------------------------------------- COMPARE PARALLEL VALUES WITH NON-PARALLEL --------------------------------------------
t_step = 0.01

u = [50,-50]


#t,y = arm1.fixed_RK_4(t_step,u)

#p_u = torch.Tensor(u).repeat(n_arms).reshape(n_arms, -1)

p_u = torch.Tensor(u).reshape(2,1).repeat(n_arms,1,1) # input must be in shape: n_arms x 2 x 1


prl_t,prl_y = paralell_arms1.fixed_RK_4(t_step,p_u)

p_u2 = torch.Tensor(u).reshape(2,1).repeat(n_arms2,1,1)
prl_t2,prl_y2 = paralell_arms2.fixed_RK_4(t_step,p_u2)


#prl_y = torch.squeeze(prl_y)

for i in range(int(tspan[1]/ t_step)+1):


    print(i)

    print(sum(np.abs(torch.mean(prl_y[i], dim=0) - torch.mean(prl_y2[i], dim=0))), '\n')
    #print(sum(np.abs(torch.mean(prl_y[i],dim=0) - torch.Tensor(y[i]))),'\n')


# for i in range(101):
#
#     print(prl_y[i,0,:] - prl_y[i,1,:] + prl_y[i,2,:] - prl_y[i,3,:] )

# ------------------------------ COMPARE PARALLEL C AND M^-1 WITH NON-PARAL --------------------------------------------------------
# theta2 = torch.Tensor([np.pi])
# p_theta2 = theta2.repeat(n_arms)
#
# M_inv = arm1.inverse_M(theta2)
# P_inv = paralell_arms.inverse_M(p_theta2)
#
# d_theta1 = torch.Tensor([2.])
# d_theta2 = torch.Tensor([3.])
#
# p_d_theta1 = d_theta1.repeat(n_arms)
# p_d_theta2 = d_theta2.repeat(n_arms)
#
#
# cc = arm1.computeC(theta2,d_theta1,d_theta2)
# p_cc = paralell_arms.computeC(p_theta2,p_d_theta1,p_d_theta2)
#
#
#
# for i in range(n_arms):
#     print(M_inv - P_inv[:,:,i])
#     print(torch.from_numpy(cc) - p_cc[:, :, i])




# ----------------------------------- EINSUM TRIAL ------------------------------------------------------
# a = torch.Tensor(np.arange(40).reshape(2,2,10))
# b = torch.Tensor(np.arange(20).reshape(2,1,10))

# c = []
# for i in range(10):
#
#     c.append([torch.dot(a[0,:,i],b[:,0,i]),torch.dot(a[1,:,i],b[:,0,i])])
#
# print(c,"\n")
# print(torch.einsum("ijk,jpk -> ik",a,b))
# print(torch.einsum("ijk,jk -> ik",a,b.reshape(2,10)).T.shape)

