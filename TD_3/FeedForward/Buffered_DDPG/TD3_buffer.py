from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
from TD_3.FeedForward.Buffered_DDPG.Van_Replay_buffer import V_Memory_B
import torch
import numpy as np



s_file = 61 # randomly generated test seeds : 35, 71, 33, 59, 61
torch.manual_seed(s_file)
file_acc = '/home/px19783/Two_joint_arm/TD_3/FeedForward/Results/Conf3_FF_DPG_Training_accur_s'+str(s_file)+'_Best_arms.pt'
file_vel = '/home/px19783/Two_joint_arm/TD_3/FeedForward/Results/Conf3_FF_DPG_Training_vel_s'+str(s_file)+'_Best_arms.pt'



#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')

episodes = 15000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100  # 0

n_arms = 10

tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
t_step = tspan[-1] / n_RK_steps
f_points = -time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
vel_weight = 0.005 #0.005
ln_rate_c = 0.001#9.99999975e-06  # 0.01 #0.001# 0.005
ln_rate_a = 7.52500026e-04 #0.0006
std = 0.0119 #0.01#0.01#0.01  # 0.2 #0.000015#0.02
max_u = 15000 # 10000
# actor_update = 2#5 #2 #5 #4 reaches 18cm distance and stops
start_a_upd = 500  # 50#500
th_conf = 0.85#8.25#0.85
th_error = 0.025
actor_upd = 2
a_size = n_parametrised_steps * 2
batch_size = 100
buffer_size = 5000


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0

target_state = torch.tensor([x_hat, y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)


training_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=n_arms)

# Initialise actor and critic
agent = Actor_NN(dev, Output_size=n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
agent.apply(agent.small_weight_init)

c_input_s = n_parametrised_steps * 2 + 2
critic_1 = Critic_NN(batch_size, dev, input_size=c_input_s, ln_rate=ln_rate_c).to(dev)

M_buffer = V_Memory_B(a_size,n_arms,dev,batch_size = batch_size, size=buffer_size)

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
training_actions = []


Q_cost = []

best_acc = 50

update = True

for ep in range(1, episodes):

    det_actions = agent(target_state)  # may need converting to numpy since it's a tensor

    exploration = (torch.randn((n_arms, a_size)) * std).to(dev)

    Q_actions = (det_actions.view(1, a_size) + exploration).detach()

    actions = Q_actions.view(-1, 2, n_parametrised_steps) * max_u

    t, thetas = training_arm.perform_reaching(t_step, actions.detach())

    rwd = training_arm.compute_rwd(thetas, x_hat, y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    acc_rwd = rwd.clone()

    weighted_adv = (rwd + vel_weight * velocity)

    # Store transition
    M_buffer.store_transition(target_state,Q_actions,weighted_adv)

    # Sample from buffer

    spl_t_state, spl_a, spl_rwd = M_buffer.sample_transition()


    Q_v = critic_1(spl_t_state, spl_a, True) # use True since by sampling from buffer, already repeated the target_states

    c_loss = critic_1.update(spl_rwd, Q_v)

    Q_cost.append(c_loss.detach())


    if ep > start_a_upd and  ep % actor_upd == 0:

        Tar_Q = critic_1(target_state, det_actions, True) # use True since by sampling from buffer, already repeated the target_states

        agent.update(Tar_Q)

    ep_rwd.append(torch.mean(torch.sqrt(acc_rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / t_print
        print_vel = sum(ep_vel) / t_print
        print_Qcost = sum(Q_cost) / t_print
        #std *= 0.999

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ", print_acc)
        print("training velocity: ", print_vel)
        print("Q loss: ", print_Qcost)


        ep_rwd = []
        ep_vel = []
        Q_cost = []
        training_acc.append(print_acc)
        training_vel.append(print_vel)
        training_confidence = []

torch.save(training_acc,file_acc)
torch.save(training_vel,file_vel)


tst_actions = (agent(target_state) * max_u).view(1, 2, -1)
test_arm = FF_Parall_Arm_model(tspan, x0, dev, n_arms=1)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())



rwd = torch.sqrt(training_arm.compute_rwd(thetas, x_hat, y_hat, f_points))
velocity = torch.sqrt(training_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)