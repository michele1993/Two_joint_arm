from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import *
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np

# Use REINFORCE to control arm reaches in a feedforward fashion with several multiple targets (e.g. 50)
# trained in parallel, i.e. 100 arms for each target on the same step of gradient descent
# steps
# Note: the employed shape for the parallelisation of multiple targets, each with multiple arms is: n_arms x n_targets
# so different targets for the same arm comes fist in batch shape, rather than going same target for all its arms and then move to the next target


torch.manual_seed(1)  # FIX SEED

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')


episodes = 250000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps -1 # use last point with no zero action
vel_weight = 0.005
ln_rate = 0.00001
std = 0.01
max_u = 15000
th_error = 0.025
n_target_p = 50
overall_n_arms = n_target_p * n_arms

training_arm = Parall_Arm_model(tspan,x0,dev, n_arms= overall_n_arms)

# Use to randomly generate targets
target_states = training_arm.circof_random_tagrget(n_target_p)


agent = Reinf_Actor_NN(std, n_arms,max_u,dev, ln_rate= ln_rate,Output_size=n_parametrised_steps*2).to(dev)
agent.apply(agent.small_weight_init)

critic = Critic_NN(n_arms,dev).to(dev)


avr_rwd = 0
avr_vel = 0
alpha = 0.01
best_acc = 50

ep_rwd = []
ep_vel = []
ep_c_loss = []

training_acc = []
training_vel = []
training_crict_loss = []



for ep in range(1,episodes):


    actions = agent(target_states,False) # CAREFUL: actions in shape: n_arms x targets x n_param_steps

    t, thetas = training_arm.perform_reaching(t_step,actions)


    rwd = training_arm.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, n_arms) # order is tagets first and then arms

    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    agent.update(weighted_adv)

    # Learn a Q function only, not used for training
    Q_v = critic(target_states,actions.view(overall_n_arms, -1),False)

    c_loss = critic.update(weighted_adv,Q_v)
    ep_c_loss.append(c_loss.detach())

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))


    if ep % t_print == 0:

        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print
        print_c_loss = torch.mean(torch.tensor(ep_c_loss))

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("Critic loss: ", print_c_loss)
        print("BEST: ", best_acc)
        print("training accuracy: ",print_acc)
        print("training velocity: ", print_vel,"\n")

        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_crict_loss.append(print_c_loss)
        #training_actions.append(torch.mean(actions.detach(), dim=0))
        if print_acc < th_error:
            break
        ep_rwd = []
        ep_vel = []
        ep_c_loss = []

torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_Actor_1.pt')
torch.save(training_acc,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_Training_accur_1.pt')
torch.save(training_vel,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_Training_vel_1.pt')
torch.save(target_states,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_TargetPoints_1.pt')
torch.save(training_crict_loss,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_TrainingCLoss_1.pt')
torch.save(critic.state_dict(),'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Multi_target/Parallel_MultiReinf_critic_1.pt')


tst_actions = (agent(target_states,True)).view(n_target_p, 2, -1)

test_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_target_p)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())

rwd = torch.sqrt(test_arm.compute_rwd(thetas, target_states[:,0:1], target_states[:,0:1], f_points))
velocity = torch.sqrt(test_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)