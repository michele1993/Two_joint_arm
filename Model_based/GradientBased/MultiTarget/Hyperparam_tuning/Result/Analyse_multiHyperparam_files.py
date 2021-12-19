import numpy as np
import torch
import matplotlib.pyplot as plt

# get samed seeds as for hyper-param search
np.random.seed(1)
seeds = np.random.choice(1000, size=4)

n_actor_lr = 5

# NOTE: The last element of each file contains the corresponding learning rate for the actor (only parameter tuned)
file = torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Model_based/GradientBased/MultiTarget/Hyperparam_tuning/Result/MT_Grad_MB_training_acc_hyperTuning_s"
                        +str(seeds[0])+"_"+str(1)+"_oneArm.pt",map_location=torch.device('cpu'))

n_eps = len(file)


data = torch.zeros((n_actor_lr,len(seeds),n_eps))



for c in range(1,n_actor_lr+1):

    e = 0
    for s in seeds:
        # NOTE: The last element of each file contains the corresponding learning rate for the actor (only parameter tuned)
        file = torch.tensor(torch.load("/Users/michelegaribbo/PycharmProjects/Two_joint_arm/Model_based/GradientBased/MultiTarget/Hyperparam_tuning/Result/MT_Grad_MB_training_acc_hyperTuning_s"
                        +str(s)+"_"+str(c)+"_oneArm.pt",map_location=torch.device('cpu')))

        data[c-1,e , :] = file
        e += 1

torch.set_printoptions(threshold=10_000)

# NOTE: The last element of each file contains the corresponding learning rate for the actor (only parameter tuned)
mean_across_seeds = torch.mean(data,dim=1)
std_across_seeds = torch.std(data,dim=1)

ln_rates = mean_across_seeds[:,-1]
mean_across_seeds = mean_across_seeds[:,:-1]
std_across_seeds = std_across_seeds[:,:-1]


eps_mean = torch.mean(mean_across_seeds,dim=1)
print(torch.vstack([ln_rates,mean_across_seeds[:,-1]]))
print(torch.vstack([ln_rates,eps_mean]))
n_eps = torch.arange(1, n_eps)

for ln in range(0,n_actor_lr):

    # NOTE: The last element of each file contains the corresponding learning rate for the actor (only parameter tuned)
    plt.plot(n_eps,mean_across_seeds[ln,:],label=str(ln_rates[ln]))
    plt.fill_between(n_eps, mean_across_seeds[ln,:] - std_across_seeds[ln,:], mean_across_seeds[ln,:] + std_across_seeds[ln,:], alpha=0.1)


plt.legend()
plt.show()