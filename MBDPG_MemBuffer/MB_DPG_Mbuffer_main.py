from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
from MB_DPG.FeedForward.Learnt_arm_model import learnt_ArmModel
from MBDPG_MemBuffer.Memory_Buffer import MemBuffer

# Best params from search: model_lr = 8.3500e-03; ln_rate_a = 5.1250e-04; std = 0.0124


s_file = 78 # randomly generated test seeds : 35, 71, 33, 59, 61
torch.manual_seed(s_file)

acc_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Results/Mbuffer_MBDPG_trialAccuracy_s'+str(s_file)+'_uniform.pt'
vel_file = '/Users/michelegaribbo/PycharmProjects/Two_joint_arm/MBDPG_MemBuffer/Results/Mbuffer_MBDPG_trialVelocity_s'+str(s_file)+'_uniform.pt'

#torch.autograd.set_detect_anomaly(True)

# dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

episodes = 5000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 10#10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05
model_lr = 8.3500e-03 #3.40000005e-03
ln_rate_a = 5.1250e-04 #1.87500002e-04
std = 0.0124
max_u = 15000
start_a_upd = 10#500
a_size = n_parametrised_steps * 2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
actor_update = 3
std_decay = 0.99#0.999
model_batch_s = 100#200
buffer_size = 5000

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)

training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)
est_arm = learnt_ArmModel(output_s = est_y_size, ln_rate= model_lr)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

M_buffer = MemBuffer(n_arms,a_size,est_y_size,dev,size=buffer_size)

# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_vel = []
ep_MLoss = []

training_acc = []
training_vel = []
training_actions = []

#torch.set_printoptions(threshold=10_000)

for ep in range(1, episodes):

    det_actions = agent(target_state)  # may need converting to numpy since it's a tensor

    exploration = (torch.randn((n_arms, 2, n_parametrised_steps)) * std).to(dev)

    actions = (det_actions.view(1, 2, -1) + exploration) #  * max_u , don't apply actual value for action fed to meta model

    simulator_actions = actions * max_u

    _, thetas = training_arm.perform_reaching(t_step, simulator_actions.detach())

    # extract two angles and two angle vel for all arms as target to the estimated model
    thetas = thetas[-1:,:,0:est_y_size].squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

    # ---- compute all the necessary gradients for chain-rule to update the actor ----
    diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy

    rwd = training_arm.compute_rwd(diff_thetas, x_hat, y_hat, f_points)
    vel = training_arm.compute_vel(diff_thetas, f_points)

    weight_rwd = torch.sum(rwd + vel * vel_weight)

    # Store transition in the buffer
    M_buffer.store(weight_rwd.detach(),actions.detach().view(n_arms,a_size),thetas)

    #Sampled from the buffer

    sampled_a, sampled_thetas = M_buffer.sample(model_batch_s)


    # ---- Update the model based on batch of sampled transitions -------

    est_y = est_arm(sampled_a) # compute y prediction based on current action

    model_loss = est_arm.update(sampled_thetas, est_y)
    ep_MLoss.append(model_loss.detach())


    # ---- Update the actor based on the actual observed transition -------

    if ep > start_a_upd: #and ep % actor_update ==0:


        # Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent

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


        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Model loss: ", print_MLoss)


        # if print_acc < th_error:
        #     break

        ep_rwd = []
        ep_vel = []
        ep_MLoss = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)

print(s_file)

# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Results/MB_DPG_FF_actor_final_s35.pt')
# torch.save(est_arm.state_dict(), '/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Results/MB_DPG_FF_model_final_s35.pt')
torch.save(training_acc,acc_file)
torch.save(training_vel,vel_file)