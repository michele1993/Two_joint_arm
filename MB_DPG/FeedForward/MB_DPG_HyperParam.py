from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
from MB_DPG.FeedForward.Learnt_arm_model import learnt_ArmModel

# ISSUE: code reproduce correctly only results for first set of params, but then already for the second
# set of params, results change, suggestting something of previous iteration, affected the result

torch.manual_seed(1)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')

episodes = 5001
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 10 #10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05 #0.2#0.4
range_ln_rate = torch.linspace(0.00001,0.01,10) #[1.0000e-05] #[5.5600e-03]#torch.linspace(0.00001,0.01,10)#0.00005
range_ln_rate_2 = torch.linspace(0.00001,0.01,10)#[1.1200e-03] #[1.0000e-05] #[4.4500e-03]
#range_std = torch.linspace(0.001,0.05,10)#0.01
#ln_rate_a = 0.001
#model_lr = 0.001
#std = 0.0124
max_u = 15000
start_a_upd = 500 # 1000 performs much worse
a_size = n_parametrised_steps *2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
actor_update = 3
std_decay = 0.999

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792 #0.396#0.792
y_hat = 0 #-0.396

#total_acc = torch.zeros(len(range_ln_rate)*len(range_std), 3)
#total_vel = torch.zeros(len(range_ln_rate)*len(range_std), 3)
total_acc = torch.zeros(len(range_ln_rate)*len(range_ln_rate), 3)
total_vel = torch.zeros(len(range_ln_rate)*len(range_ln_rate), 3)

i = 0 # initialise counter across all loops
target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)

#params_trials = [] # DELETE ME


for model_lr  in range_ln_rate: #ln_rate_a
    #for std in range_std:
    for ln_rate_a in range_ln_rate_2: #range_ln_rate


        torch.manual_seed(1) # re-set seeds everytime to ensure same initialisation

        training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)
        est_arm = learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr).to(dev)

        # Initialise actor and critic
        agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
        agent.apply(agent.small_weight_init)

        # DELETE ME:
        # params_trials.append(est_arm.l1.weight.clone())
        # if len(params_trials)>1:
        #     print(torch.eq(params_trials[i],params_trials[i-1]))


        ep_rwd = []
        ep_vel = []
        ep_MLoss = []

        training_acc = None
        training_vel = None

        rwd = torch.zeros(1)
        vel = torch.zeros(1)

        std = 0.0124 # Need to reset it!!!!

        for ep in range(1, episodes):

            det_actions = agent(target_state)  # may need converting to numpy since it's a tensor

            exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)

            actions = (det_actions.view(1, 2, -1) + exploration) #  * max_u , don't apply actual value for action fed to meta model

            simulator_actions = actions * max_u

            t, thetas = training_arm.perform_reaching(t_step, simulator_actions.detach())

            # extract two angles and two angle vel for all arms as target to the estimated model
            thetas = thetas[-1:,:,0:est_y_size].squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

            # Update the estimated model:
            est_y = est_arm(actions.view(n_arms,a_size).detach()) # compute y prediction based on current action

            model_loss = est_arm.update(thetas, est_y)
            ep_MLoss.append(model_loss.detach())

            # compute all the necessary gradients for chain-rule update of actor
            diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy

            rwd = training_arm.compute_rwd(diff_thetas, x_hat, y_hat, f_points)
            vel = training_arm.compute_vel(diff_thetas, f_points)

            if ep > start_a_upd and ep % actor_update == 0:

                # Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent
                weight_rwd = torch.sum(rwd + vel * vel_weight)

                # compute gradient of rwd with respect to outcome
                dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs = diff_thetas)[0]

                est_y = est_arm(actions.view(n_arms, a_size)) # re-estimate values since model has been updated

                # compute gradient of rwd with respect to actions, using environment outcome
                dr_da = torch.autograd.grad(outputs= est_y, inputs = actions, grad_outputs= dr_dy.squeeze())[0]

                agent.MB_update(actions,dr_da)


            # Estimate for accuracy only
            ep_rwd.append(torch.mean(torch.sqrt(rwd.detach())))
            ep_vel.append(torch.mean(torch.sqrt(vel.detach())))

            if ep % 10 == 0:  # decays works better if applied every 10 eps
                std *= std_decay

            if ep % t_print == 0:

                print_acc = sum(ep_rwd) / t_print
                print_vel = sum(ep_vel) / t_print
                print_MLoss = sum(ep_MLoss) / t_print

                print("episode: ", ep)
                print("training accuracy: ", print_acc)
                print("training velocity: ", print_vel)
                print("Model loss: ", print_MLoss)

                ep_rwd = []
                ep_vel = []
                ep_MLoss = []
                training_acc = print_acc.clone()
                training_vel = print_vel.clone()

        total_acc[i,:] = torch.tensor([training_acc, model_lr, ln_rate_a])
        total_vel[i,:] = torch.tensor([training_vel, model_lr, ln_rate_a])
        i +=1
        print("iteration n: ", i)

print("gpu")
#print(total_acc)
torch.save(total_acc,'/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Results/HyperParam_search/MB_DPG_FF_HyperParameter_acc_s1_gpu.pt')
torch.save(total_vel,'/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Results/HyperParam_search/MB_DPG_FF_HyperParameter_vel_s1_gpu.pt')