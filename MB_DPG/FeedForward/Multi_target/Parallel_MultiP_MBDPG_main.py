from MB_DPG.FeedForward.Multi_target.MultiP_learnArm_model import Multi_learnt_ArmModel
from TD_3.FeedForward.FF_AC import *
from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
import torch
import numpy as np

# best params so far: ln_rate_a = 4.75000015e-05; model_lr = 5.40000014e-03; std = 0.0124 with decay

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# hyperparam search based on seeds: [ 42, 245, 918]
# so far using fixed std, with no decay worked best, saved as ..._2
seed_v = 528 # test seeds: [4, 418, 81,528,702]
torch.manual_seed(seed_v)  # 16 FIX SEED

accuracy_file = '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_training_acc_test_s'+str(seed_v)+'_2.pt'
actor_file = '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_actor_test_s'+str(seed_v)+'_2.pt'
#TarPoints_file = '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_test_s'+str(seed_v)+'.pt'
model_file = '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_model_test_s'+str(seed_v)+'_2.pt'
# velocity_file = '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_training_vel_test_s'+str(seed_v)+'.pt'


episodes = 35000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 10 # n. of arms for each target
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05#0.2#0.4
ln_rate_a = 4.75000015e-05 #1.87500002e-04#working well: 0.00005
model_lr = 5.40000014e-03 #3.40000005e-03 # working well: 0.001
std = 0.0124 #working well: 0.01
max_u = 15000
start_a_upd = 500 # 1000 performs much worse
a_size = n_parametrised_steps *2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
actor_update = 3
std_decay = 0.999# 0.9999 with decay every 100 steps was working good
n_target_p = 50
overall_n_arms = n_target_p * n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)


training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=overall_n_arms)
est_arm = Multi_learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr).to(dev)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

# Use to randomly generate targets in front of the arm and on the max distance circumference
#target_states = training_arm.circof_random_tagrget(n_target_p)
target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt')


# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_vel = []
ep_MLoss = []

training_acc = []
training_vel = []
training_actions = []

rwd = torch.zeros(1)
vel = torch.zeros(1)


for ep in range(1, episodes):

    det_actions = agent(target_states)  # may need converting to numpy since it's a tensor

    # add noise to each action for each arm for each target
    exploration = (torch.randn((overall_n_arms, 2, n_parametrised_steps)) * std).to(dev)

    # need to repeat the deterministic action for each arm so can add noise to each
    actions = det_actions.repeat(n_arms,1).view(overall_n_arms, 2, n_parametrised_steps) + exploration # need to repeat action means so that can add noise for each target x arm cobination

    simulator_actions = actions * max_u

    t, thetas = training_arm.perform_reaching(t_step, simulator_actions.detach())

    # extract two angles and two angle vel for all arms as target to the estimated model
    thetas = thetas[-1:,:,0:est_y_size]#.squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

    # Update the estimated model:
    est_y = est_arm(actions.view(overall_n_arms,a_size).detach()) # compute y prediction based on current action

    model_loss = est_arm.update(thetas.squeeze(dim=-1), est_y)
    ep_MLoss.append(model_loss)

    # compute all the necessary gradients for chain-rule update of actor
    diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy with autograd.grad


    # Change this (?)
    rwd = training_arm.multiP_compute_rwd(diff_thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    vel = training_arm.compute_vel(diff_thetas, f_points)

    if ep > start_a_upd and ep % actor_update ==0:

        # Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent
        weight_rwd = torch.sum(rwd + vel * vel_weight)

        # compute gradient of rwd with respect to outcome
        dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs = diff_thetas)[0]


        est_y = est_arm(actions.view(overall_n_arms, a_size)) # re-estimate model prediction since model has been updated and gradient changed

        # compute gradient of rwd with respect to actions, using environment outcome
        dr_da = torch.autograd.grad(outputs= est_y, inputs = actions, grad_outputs= dr_dy.squeeze())[0]

        agent.MB_update(actions,dr_da)


    # Estimate for accuracy only
    ep_rwd.append(torch.mean(torch.sqrt(rwd.detach())))
    ep_vel.append(torch.mean(torch.sqrt(vel.detach())))

    if ep % 100 == 0: # 10 decay std
        std *= std_decay

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        print_MLoss = sum(ep_MLoss) / t_print


        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Model loss: ", print_MLoss)


        ep_rwd = []
        ep_vel = []
        ep_MLoss = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)

torch.save(training_acc,accuracy_file)
torch.save(agent.state_dict(), actor_file)
# torch.save(target_states,TarPoints_file)
torch.save(est_arm.state_dict(), model_file)
# torch.save(training_vel,velocity_file)