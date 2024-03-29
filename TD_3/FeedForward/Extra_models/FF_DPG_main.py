from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.Extra_models.FF_AC_DPG_MB import *
from TD_3.FeedForward.Buffered_DDPG.Van_Replay_buffer import V_Memory_B
import torch
import numpy as np
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Implement DPG for the FeedForward arm model, inputting the desired location to a actor NN,
# which outputs entire sequence of actions, then train a Q to predict simple advantage on entire
# trajectory, and differentiate through that train actor

torch.manual_seed(0) # FIX SEED

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')

episodes = 20000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 1
batch_size = 100
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps -1 # use last point with no zero actions
vel_weight = 0.005#0.05#0.8
ln_rate_c = 0.005
ln_rate_a = 0.000001#0.000005
std = 0.05#0.03 # 0.001 doesn't work, not enough exploration
max_u = 15000 #15000
actor_update = 1#10#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd = 100


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev) #.repeat(n_arms,1).to(dev)
training_arm = FF_Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)

#Initialise actor and critic
agent = Actor_NN(dev,Output_size = n_parametrised_steps *2,ln_rate = ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps *2
critic_1 = Critic_NN(n_arms, dev,a_size = c_input_s,ln_rate=ln_rate_c).to(dev)
#critic_2 = Critic_NN(input_size= c_input_s, ln_rate=ln_rate_c).to(dev)

m_buffer = V_Memory_B(n_parametrised_steps,n_arms,dev,batch_size = batch_size,size = 5000*4)

best_acc = 50

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
training_actions = []
training_confidence = []

Tar_Q = torch.zeros(1)
Q_v = torch.zeros(1)

for ep in range(1,episodes):


    det_actions = agent(target_state) # may need converting to numpy since it's a tensor

    exploration = (torch.randn((n_arms,2,n_parametrised_steps)) * std).to(dev)

    Q_actions = (det_actions.view(1,2,-1) + exploration).detach()

    zero_actions = torch.zeros(n_arms, 2, time_window_steps).to(dev)
    actions = torch.cat([Q_actions * max_u, zero_actions], dim=2)


    t, thetas = training_arm.perform_reaching(t_step,actions.detach())

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    m_buffer.store_transition(target_state, Q_actions, weighted_adv)

    # mean_G = torch.mean(torch.mean(weighted_adv, dim=0), dim=0)
    # std_G = torch.sqrt(torch.sum((mean_G - weighted_adv) ** 2) / (n_arms - 1))


    if ep > start_a_upd and ep % t_print == 0:

        for i in range(t_print):
            spl_t_state, spl_a, spl_adv = m_buffer.sample_transition()

            Q_v = critic_1(spl_t_state, spl_a.view(batch_size, -1), True) #BE CAREFUL WHEN HAVE MULTIPLE TARGET STATES as need to repeat in AC for each
            c_loss = critic_1.update(spl_adv, Q_v)

            t_state = spl_t_state[0, :].view(1, 2)  # NOTE: need to be adpated when using mutiple end-points
            det_actions = agent(t_state)
            Tar_Q = critic_1(t_state, det_actions, True)  # want to max the advantage

            if i % actor_update == 0:
                agent.update(Tar_Q)


    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        #std *= 0.9#9
        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print
        print_conf = sum(training_confidence) / t_print

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Confidence: ", print_conf, "\n")
        print("Target Q: ", torch.mean(Tar_Q.detach()))
        print("Current Q:", torch.mean(Q_v.detach()),"\n")
        ep_rwd = []
        ep_vel = []
        training_confidence = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_actions.append(torch.mean(det_actions.detach(),dim=0))

tst_actions = (agent(target_state) * max_u).view(1, 2, -1)
test_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=1)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())



rwd = torch.sqrt(training_arm.compute_rwd(thetas, x_hat, y_hat, f_points))
velocity = torch.sqrt(training_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)

# torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Actor_2.pt')
# torch.save(critic_1.state_dict(), '/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Critic_2.pt')
# torch.save(training_acc,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_accur_2.pt')
# torch.save(training_vel,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_vel_2.pt')
# torch.save(training_actions,'/home/px19783/Two_joint_arm/TD_3/FeedForward/FF_DPG_Training_actions_2.pt')

