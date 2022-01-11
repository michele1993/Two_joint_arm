from MB_DPG.FeedForward.Multi_target.MultiP_learnArm_model import Multi_learnt_ArmModel
from TD_3.FeedForward.FF_AC import *
from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from MBDPG_MemBuffer.Multi_target.MultiT_Memory_Buffer import MemBuffer
import torch
import numpy as np
import argparse

# Note the directories and training are set-up to add an offset to the outcome, to test model resistance to bias, change
# for normal training (i.e. change saving directories and set  offset_noise = 0)

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?', default=1)
# Default values represent best values from hyperparam search:
parser.add_argument('--modelLr',    '-m', type=float, nargs='?', default=5.40000014e-03)
parser.add_argument('--actorLr',   '-a', type=float, nargs='?', default= 5.2500e-05)
parser.add_argument('--counter',   '-i', type=int, nargs='?')
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)

args = parser.parse_args()
seed = args.seed
i = args.counter
model_ln = args.modelLr
actor_ln = args.actorLr
search_hyperParam = args.hyperparam_search


# best params so far: ln_rate_a = 5.2500e-05; model_lr = 5.40000014e-03; std = 0.0124 with decay
#_3 no std decay, works worse; _2  std decay: 0.999 works best; otherwise: 0.99

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')


torch.manual_seed(seed)  # 16 FIX SEED


hyper_tuning = search_hyperParam

if not hyper_tuning:

    accuracy_file = '/user/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Multi_target/Result/Target_offset/MultiPMB_MbufferDPG_training_acc_test_s'+str(seed)+'_'+str(i)+'_oneArm_offsetMedium.pt'
    actor_file = '/user/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Multi_target/Result/Target_offset/MultiPMB_MbufferDPG_actor_test_s'+str(seed)+'_'+str(i)+'_oneArm_offsetMedium.pt'
    model_file = '/user/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Multi_target/Result/Target_offset/MultiPMB_MbufferDPG_model_test_s'+str(seed)+'_'+str(i)+'_oneArm_offsetMedium.pt'
    modelLoss_file = '/user/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Multi_target/Result/Target_offset/MultiPMB_MbufferDPG_model_loss_test_s'+str(seed)+'_'+str(i)+'_oneArm_offsetMedium.pt'

    episodes = 20001
else:

    accuracy_file = '/user/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Multi_target/Hyperparam_tuning/Result/MultiPMB_MbufferDPG_training_acc_hyperTuning_s' + str(
        seed) + '_' + str(i) + '_oneArm.pt'
    episodes = 12001


n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 1#10 # n. of arms for each target
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05#0.2#0.4
ln_rate_a = actor_ln #4.75000015e-05 #4.75000015e-05 #1.87500002e-04#working well: 0.00005
model_lr = model_ln #5.40000014e-03 #3.40000005e-03 # working well: 0.001
std = 0.0124 #working well: 0.01
max_u = 15000
start_a_upd = 100#10#500 # 1000 performs much worse
a_size = n_parametrised_steps *2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
n_target_p = 50
overall_n_arms = n_target_p * n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)
std_decay = 0.999 # work worse: 0.99 and 1
model_batch_s = 3000#200
buffer_size = 15000 #25000 performs worse
#action_noise = 0.1
#state_noise = 0.1 # 0.25
offset_noise = 0.125 # low: 0.05, Medium: 0.125, high: 0.25 (saved simply as offset)

training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=overall_n_arms)
est_arm = Multi_learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr).to(dev)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

M_buffer = MemBuffer(overall_n_arms,a_size,est_y_size,dev,size=buffer_size)


# Use to randomly generate targets in front of the arm and on the max distance circumference
#target_states = training_arm.circof_random_tagrget(n_target_p)
# Use same (initially randomly chosen) target points for all algorithms
#target_states = torch.load('/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))
target_states = torch.load('/user/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt',map_location=torch.device('cpu'))


# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_vel = []
ep_MLoss = []

training_acc = []
training_vel = []
training_modelLoss = []

# ------ ADD a fixed offset to states for each target (i.e. induce an artificial bias to model to show MB-DPG more resistent) -----
## Wrong way to compute offset: offset = torch.randn(model_batch_s,est_y_size) * offset_noise

#offset = torch.randn(n_target_p,est_y_size) * offset_noise # a different fixed offset for each target
offset = (torch.randn(1,est_y_size) * offset_noise).repeat(n_target_p,1) # same fixed offset for each target



for ep in range(1, episodes):


    det_actions = agent(target_states)  # may need converting to numpy since it's a tensor

    # add noise to each action for each arm for each target
    exploration = (torch.randn((overall_n_arms, 2, n_parametrised_steps)) * std).to(dev)

    # need to repeat the deterministic action for each arm so can add noise for each target x arm combination
    actions = det_actions.repeat(n_arms,1).view(overall_n_arms, 2, n_parametrised_steps) + exploration

    simulator_actions = actions * max_u # actor outputs tanh, so multiply by max strength

    _, thetas = training_arm.perform_reaching(t_step, simulator_actions.detach())


    # extract two angles and two angle vel for all arms as target to the estimated model
    thetas = thetas[-1:,:,0:est_y_size]#.squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel


    # ------- Compute (differentiable) reward ---------------------

    diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy with autograd.grad later

    rwd = training_arm.multiP_compute_rwd(diff_thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms)
    vel = training_arm.compute_vel(diff_thetas, f_points)

    weight_rwd = torch.sum(rwd + vel * vel_weight)

    # ------- Stores actions and outcomes in buffer -------------

    # Add offset here, so that only used to update model (i.e. to induce a bias in the model):
    M_buffer.store(actions.detach().view(overall_n_arms,a_size),thetas.squeeze() + offset) # Store transition in the buffer


    # --------- Sampled from the buffer ---------- :

    sampled_a, sampled_thetas = M_buffer.sample(model_batch_s)

    # Try to Normalise states to be learnt, no working, I guess because of change of magnitude of derivatives in actor update (normalise observation ?)
    #sampled_thetas = sampled_thetas  / torch.sum(sampled_thetas,dim=0, keepdim=True)

    # ---- Update the model based on batch of sampled transitions -------
    est_y = est_arm(sampled_a) # compute y prediction based on sampled action
    model_loss = est_arm.update(sampled_thetas, est_y)
    ep_MLoss.append(model_loss.detach())


    if ep > start_a_upd: # After model pre-trained:

        # ---- Update the actor based on the actual observed outcome
        # compute all the necessary gradients for chain-rule to update the actor ----

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
        training_modelLoss.append(print_MLoss)

if not hyper_tuning:
    torch.save(training_acc, accuracy_file)
    #torch.save(agent.state_dict(), actor_file)
    #torch.save(est_arm.state_dict(), model_file)
    torch.save(training_modelLoss,modelLoss_file)

else:
    training_acc.append(actor_ln)
    torch.save(training_acc, accuracy_file)

