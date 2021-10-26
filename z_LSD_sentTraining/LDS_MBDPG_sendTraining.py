from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from MBDPG_MemBuffer.Linear_DynamicalSystem.SingleTarget.Linear_DS_agent import Linear_DS_agent
import torch
import numpy as np
from MB_DPG.FeedForward.Learnt_arm_model import learnt_ArmModel
from MBDPG_MemBuffer.Memory_Buffer import MemBuffer
import argparse

# trial inputs: -s 0 -m 0.0034000000450760126 -a 0.0001 -i 1

parser = argparse.ArgumentParser()
parser.add_argument('--seed',    '-s', type=int, nargs='?')
# Default values represent best values from hyperparam search:
parser.add_argument('--modelLr',    '-m', type=float, nargs='?',default= 5.2500e-03) #5.2500e-03
parser.add_argument('--actorLr',   '-a', type=float, nargs='?', default= 1.0000e-04) #1.0000e-04
parser.add_argument('--std',   '-d', type=float, nargs='?', default= 0.0124) # for max = 10000, best std=  0.0192
parser.add_argument('--counter',   '-i', type=int, nargs='?')
parser.add_argument('--hyperparam_search',   '-hs', type=bool, nargs='?', default=False)


args = parser.parse_args()
seed = args.seed
i = args.counter
model_ln = args.modelLr
actor_ln = args.actorLr
search_hyperParam = args.hyperparam_search
std = args.std

torch.manual_seed(seed)
dev = torch.device('cpu')

if not search_hyperParam:

    accuracy_file = '/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/SingleTarget/Result/LDS_MBDPG_training_acc_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim_2.pt'
    actor_file = '/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/SingleTarget/Result/LDS_MBDPG_actor_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim.pt'
    model_file = '/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/SingleTarget/Result/LDS_MBDPG_model_test_s'+str(seed)+'_'+str(i)+'_oneArmOptim.pt'
    episodes = 15001
else:

    accuracy_file = '/home/px19783/Two_joint_arm/MBDPG_MemBuffer/Linear_DynamicalSystem/SingleTarget/Hyperparam_tuning/Result/LDS_MBDPG_training_acc_hyperTuning_s' + str(
        seed) + '_' + str(i) + '_oneArm_std.pt'
    episodes = 10001

print("a: ",actor_ln, "m: ",model_ln, "h ", search_hyperParam,"e ",episodes, "d ",std)

n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 1#0#10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.05
#std = 0.0124 * 3/2 #2/3 # since max control signal was reduced of 1/3
max_u = 15000 #10000
start_a_upd = 100
a_size = n_parametrised_steps * 2
est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
std_decay = 0.999 #0.99
model_batch_s = 100
buffer_size = 500

# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)

training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)
est_arm = learnt_ArmModel(output_s = est_y_size, ln_rate= model_ln)

# Initialise actor and critic
agent = Linear_DS_agent(t_step,n_parametrised_steps,actor_ln,dev).to(dev)
agent.apply(agent.small_weight_init)

M_buffer = MemBuffer(n_arms,a_size,est_y_size,dev,size=buffer_size)

# Initialise some useful variables
best_acc = 50

ep_rwd = []
ep_vel = []
ep_MLoss = []

training_acc = []
training_vel = []

#torch.set_printoptions(threshold=10_000)


for ep in range(1, episodes):

    det_actions,_ = agent(target_state)  # may need converting to numpy since it's a tensor

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
    M_buffer.store(actions.detach().view(n_arms,a_size),thetas) # weight_rwd.detach(),

    #Sampled from the buffer

    sampled_a, sampled_thetas = M_buffer.sample(model_batch_s)


    # ---- Update the model based on batch of sampled transitions -------

    est_y = est_arm(sampled_a) # compute y prediction based on current action

    model_loss = est_arm.update(sampled_thetas, est_y)
    ep_MLoss.append(model_loss.detach())


    # ---- Update the actor based on the actual observed transition -------

    if ep > start_a_upd: #and ep % actor_update ==0:

        # ---- Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent


        # compute gradient of rwd with respect to outcome
        dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs = diff_thetas)[0]

        est_y = est_arm(actions.view(n_arms, a_size)) # re-estimate values since model has been updated


        # compute gradient of rwd with respect to actions, using environment outcome
        dr_da = torch.autograd.grad(outputs= est_y, inputs = actions, grad_outputs= dr_dy.squeeze(0))[0]

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
        #print("Model loss: ", print_MLoss)


        ep_rwd = []
        ep_vel = []
        ep_MLoss = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)

if not search_hyperParam:
    torch.save(training_acc, accuracy_file)
    #torch.save(agent.state_dict(), actor_file)
    #torch.save(est_arm.state_dict(), model_file)

else:
    training_acc.append(model_ln)
    training_acc.append(actor_ln)
    torch.save(training_acc, accuracy_file)