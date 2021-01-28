import torch
import numpy as np
import matplotlib.pyplot as plt

# _1 & _2 : cross_grid on thor and u regularised
# _3 : cross_grid on thor and d_thor regularised
# _4: regularisation of thor only over large range, linspace(0,0.001,1000), this doesn't seem to have any effect on reducing euclidean acceleration

training_acc = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_training_accuracy_4.pt')
training_vel = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_training_velocity_4.pt')
HyperParams1 = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_hyparamW1_4.pt')
#HyperParams2 = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_hyparamW2_3.pt')
accelleration = torch.load('/home/px19783/PycharmProjects/Two_joint_arm/Supervised_learning/Feed_Back/FB_L_Decay/Regularised/Parallel_HyperParams/Results/Spvsd_FB_Parallel_final_accelleration_4.pt')




HyperParams1 = torch.squeeze(HyperParams1).cpu()
#HyperParams2 = torch.squeeze(HyperParams2).cpu()


training_acc = torch.stack(training_acc).cpu()
training_vel = torch.stack(training_vel).cpu()

# Use this to get all the models ---------------------------------------------------------------
# final_acc = training_acc[-1,:]
# final_vel = training_vel[-1,:]
#final_accel = torch.squeeze(accelleration[:,:100]).cpu()
# ---------------------------------------------------------------------------------------------

# Only Get index of models that achieved good accuracy on final training episode ------------------
accuracy_thrld = 0.01
indx = training_acc[-1,:] < accuracy_thrld
final_acc = training_acc[-1,indx]
final_vel = training_vel[-1,indx]
final_accel = torch.squeeze(accelleration[:,indx]).cpu()
# ---------------------------------------------------------------------------------------------

n_models = np.arange(0,torch.sum(torch.squeeze(indx))) # count the number of models passing the threshold
t = np.linspace(0,0.4,100)


ep = np.arange(0,len(training_acc))


fig1 = plt.figure()

ax1 = fig1.add_subplot(311)

ax1.scatter(n_models,final_acc)

ax2= fig1.add_subplot(312)

ax2.scatter(n_models,final_vel)

ax3= fig1.add_subplot(313)

ax3.plot(t, final_accel)

#ax3.plot(t,final_accel[:,2])
#l1,l2,l3,l4,l5 = ax3.plot(t,final_accel)
#ax3.legend((l1,l2,l3,l4,l5),('hp1', 'hp2','hp3','hp4','hp5'))



plt.show()
