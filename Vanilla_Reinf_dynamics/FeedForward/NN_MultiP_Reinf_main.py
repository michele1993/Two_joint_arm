from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import Reinf_Actor_NN
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np

# Use REINFORCE to control arm reaches in a feedforward fashion
torch.manual_seed(16)  # FIX SEED

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')


episodes = 120000
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


# Target endpoint, based on matlab - reach straight in front, at shoulder height
# x_hat = 0.792
# y_hat = 0
# target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)
n_target_p = 3

x_hat_1 = 0.792
y_hat_1 = 0

x_hat_2 = 0.396
y_hat_2 = 0.686

x_hat_3 = 0.396
y_hat_3 = -0.686

target_points = torch.tensor([[x_hat_1, y_hat_1],[x_hat_2, y_hat_2], [x_hat_3, y_hat_3]])

training_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)
agent = Reinf_Actor_NN(std, n_arms,max_u,dev, ln_rate= ln_rate,Output_size=n_parametrised_steps*2).to(dev)
agent.apply(agent.small_weight_init)


avr_rwd = 0
avr_vel = 0
alpha = 0.01
best_acc = 50

ep_rwd = []
ep_vel = []

training_acc = []
training_vel = []
#training_actions = []


for ep in range(1,episodes):

    trg_idx = ep % n_target_p
    target_state = target_points[trg_idx:trg_idx+1,:]

    actions = agent(target_state,False) # may need converting to numpy since it's a tensor

    t, thetas = training_arm.perform_reaching(t_step,actions)

    rwd = training_arm.compute_rwd(thetas,target_state[0,0],target_state[0,1], f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))


    if ep % t_print == 0:

        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("episode: ", ep)
        print("training accuracy: ",print_acc)
        print("training velocity: ", print_vel)

        training_acc.append(print_acc)
        training_vel.append(print_vel)
        #training_actions.append(torch.mean(actions.detach(), dim=0))
        if print_acc < th_error:
            break
        ep_rwd = []
        ep_vel = []

torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/NN_FF_MultiReinf_Actor_1.pt')
torch.save(training_acc,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/NN_FF_MultiReinf_Training_accur_1.pt')
torch.save(training_vel,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/NN_FF_MultiReinf_Training_vel_1.pt')
torch.save(target_points,'/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/NN_FF_MultiReinf_TargetPoints_1.pt')

tst_actions = (agent(target_state,True)).view(1, 2, -1)
test_arm = Parall_Arm_model(tspan,x0,dev, n_arms=1)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())



rwd = torch.sqrt(training_arm.compute_rwd(thetas, target_state[0,0], target_state[0,1], f_points))
velocity = torch.sqrt(training_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)