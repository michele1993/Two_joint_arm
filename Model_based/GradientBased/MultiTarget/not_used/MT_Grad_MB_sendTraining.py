from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
from MB_DPG.FeedForward.Multi_target.MultiP_learnArm_model import Multi_learnt_ArmModel
from MB_DPG.FeedForward.Learnt_arm_model import learnt_ArmModel
from MBDPG_MemBuffer.Multi_target.MultiT_Memory_Buffer import MemBuffer
import argparse

# trial inputs: -s 0 -m 0.0034000000450760126 -a 0.0001 -i 1

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?')
# Default values represent best values from hyperparam search:
parser.add_argument('--modelLr',    '-m', type=float, nargs='?',default= 5.0500e-03) #5.2500e-03
parser.add_argument('--actorLr',   '-a', type=float, nargs='?', default= 8.7500e-05) #8.7500e-05  #5.2500e-04
parser.add_argument('--std',   '-d', type=float, nargs='?', default= 0.0124) #0.0124 # for max = 10000, best std=  0.0192
parser.add_argument('--counter',   '-i', type=int, nargs='?',default= 0)
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)


args = parser.parse_args()
seed = args.seed
i = args.counter
model_lr = args.modelLr
ln_rate_a = args.actorLr
search_hyperParam = args.hyperparam_search
std = args.std

torch.manual_seed(seed)
dev = torch.device('cpu')

if not search_hyperParam:


    accuracy_file = '/user/home/px19783/Two_joint_arm/Model_based/GradientBased/MultiTarget/Results/MT_Grad_MB_training_acc_test_s' + str(seed) + '_' + str(i) + '_oneArmOptim.pt'
    actor_file = '/user/home/px19783/Two_joint_arm/Model_based/GradientBased/MultiTarget/Results/MT_Grad_MB_actor_test_s' + str(seed) + '_' + str(i) + '_oneArmOptim.pt'
    model_file = '/user/home/px19783/Two_joint_arm/Model_based/GradientBased/MultiTarget/Results/MT_Grad_MB_model_test_s' + str(seed) + '_' + str(i) + '_oneArmOptim.pt'
    episodes = 60001


else:

    accuracy_file = '/user/home/px19783/Two_joint_arm/Model_based/GradientBased/MultiTarget/Hyperparam_tuning/Result/MT_Grad_MB_training_acc_hyperTuning_s' + str(
        seed) + '_' + str(i) + '_oneArm_std.pt'
    episodes = 10001


n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 10#0#10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05
max_u = 15000 #10000
start_a_upd = 100
a_size = n_parametrised_steps * 2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
std_decay = 0.999#0.999 #0.99
batch_size = 100 #3000
buffer_size = 500 #15000#15000
n_targets = 2 #50
overall_n_traj = n_targets * n_arms

training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=overall_n_traj)

# CHANGEEEEE!!!!
target_states = training_arm.circof_random_tagrget(n_targets)
#target_states = torch.load('/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))
#target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))

#est_arm = Multi_learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr).to(dev)
est_arm = learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr).to(dev)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)


M_buffer = MemBuffer(overall_n_traj,a_size,est_y_size,dev,size=buffer_size)

# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_est_rwd = []
ep_vel = []
ep_MLoss = []

training_acc = []
training_vel = []




for ep in range(1, episodes):

    det_actions = agent(target_states)  # may need converting to numpy since it's a tensor

    exploration = (torch.randn((overall_n_traj, 2, n_parametrised_steps)) * std).to(dev)

    actions = det_actions.repeat(n_arms,1).view(overall_n_traj,2,n_parametrised_steps) + exploration #  * max_u , don't apply actual value for action fed to meta model

    simulator_actions = actions * max_u

    _, thetas = training_arm.perform_reaching(t_step, simulator_actions.detach())

    # extract two angles and two angle vel for all arms as target to the estimated model
    thetas = thetas[-1:,:,0:est_y_size]#.squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

    # ---- compute all the necessary gradients for chain-rule to update the actor ----

    rwd = training_arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    vel = training_arm.compute_vel(thetas, f_points)

    weight_rwd = torch.sum(rwd + vel * vel_weight)


    # Store transition in the buffer
    M_buffer.store(actions.detach().view(overall_n_traj,a_size),thetas.squeeze()) # weight_rwd.detach()


    #Sampled from the buffer
    sampled_a, sampled_thetas = M_buffer.sample(batch_size)


    sampled_thetas = sampled_thetas #+ torch.randn((batch_size,est_y_size))


    # ---- Update the model based on batch of sampled transitions -------

    # Try adding some smoothing noise to action, like it TD3 (not helping)
    # noisy_actions = torch.clip(sampled_a + torch.randn((batch_size,a_size)) *0.1,-1,1)
    # est_y = est_arm(noisy_actions) # compute y prediction based on current action

    est_y = est_arm(sampled_a)

    model_loss = est_arm.update(sampled_thetas, est_y)
    ep_MLoss.append(model_loss.detach())



    # ---- Update the actor based on the actual observed transition -------

    if ep > start_a_upd: #and ep % actor_update ==0:

        est_y = est_arm(actions.view(overall_n_traj,a_size))

        #ep_MLoss.append(torch.mean((est_y - thetas.squeeze()) ** 2,dim=1))


        # Compute the estimated reward and velocity
        est_rwd = training_arm.multiP_compute_rwd(est_y.unsqueeze(0), target_states[:, 0:1], target_states[:, 1:2], f_points, n_arms)
        est_vel = training_arm.compute_vel(est_y.unsqueeze(0), f_points)

        weight_rwd = torch.sum(est_rwd + est_vel * vel_weight)

        # compute gradient of rwd with respect to outcome
        dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs = est_y)[0]


        # compute gradient of rwd with respect to actions, using environment outcome
        dr_da = torch.autograd.grad(outputs= est_y, inputs = actions, grad_outputs= dr_dy.squeeze())[0]

        agent.MB_update(actions,dr_da)
        ep_est_rwd.append(torch.mean(torch.sqrt(est_rwd)))


    # Estimate for accuracy only
    ep_rwd.append(torch.mean(torch.sqrt(rwd.detach())))
    ep_vel.append(torch.mean(torch.sqrt(vel.detach())))

    if ep % 100 == 0:  # decays works better if applied every 10 eps
        std *= std_decay

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        print_MLoss = sum(ep_MLoss) / t_print
        print_est_acc = sum(ep_est_rwd) / t_print


        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        #print("training velocity: ", print_vel)
        print("Model loss: ", print_MLoss)
        print("Estimate acc", print_est_acc,"\n")

        ep_rwd = []
        ep_vel = []
        ep_MLoss = []
        ep_est_rwd = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)


if not search_hyperParam:

    torch.save(training_acc, accuracy_file)
    torch.save(agent.state_dict(), actor_file)
    torch.save(est_arm.state_dict(), model_file)


else:
    training_acc.append(model_lr)
    training_acc.append(ln_rate_a)
    torch.save(training_acc, accuracy_file)