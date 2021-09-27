from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import Reinf_Actor_NN
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np

# Seeds for hyperparam search: [37, 12, 72,  9, 75]
# Best params (after search) ln_rate = 0.001, std = 0.01325

s_file = 767 # randomly generated test seeds : 35, 71, 33, 59, 61

# Use REINFORCE to control arm reaches in a feedforward fashion
torch.manual_seed(s_file)

acc_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_Reinf_Training_accur_s'+str(s_file)+'_final_10arms.pt'
vel_file = '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_Reinf_Training_vel_s'+str(s_file)+'_final_10arms.pt'



#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dev = torch.device('cpu')


episodes = 5001#15000
n_RK_steps = 99
time_window_steps = 0
n_parametrised_steps = n_RK_steps - time_window_steps
t_print = 100
n_arms = 1 #50#10
tspan = [0, 0.4]
x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
t_step = tspan[-1]/n_RK_steps # torch.Tensor([tspan[-1]/n_RK_steps]).to(dev)
f_points = - time_window_steps -1 # use last point with no zero action
vel_weight = 0.005
ln_rate = 1.00000005e-03 # best for 10 arms: 0.001
std = 1.73333343e-02 # best for 10 arms: 0.01325
max_u = 15000
th_error = 0.025
std_decay = 0.999


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0
target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)


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


    actions = agent(target_state,False) # may need converting to numpy since it's a tensor

    t, thetas = training_arm.perform_reaching(t_step,actions)

    rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
    velocity = training_arm.compute_vel(thetas, f_points)

    weighted_adv = rwd + vel_weight * velocity

    agent.update(weighted_adv)

    ep_rwd.append(torch.mean(torch.sqrt(rwd)))
    ep_vel.append(torch.mean(torch.sqrt(velocity)))

    if ep % 10 == 0:  # decays works better if applied every 10 eps
        std *= std_decay


    if ep % t_print == 0:

        print_acc = sum(ep_rwd)/t_print
        print_vel = sum(ep_vel)/t_print

        if print_acc < best_acc:
            best_acc = print_acc

        print("episode: ", ep)
        print("BEST: ", best_acc)
        print("training accuracy: ",print_acc)
        print("training velocity: ", print_vel,"\n")

        training_acc.append(print_acc)
        training_vel.append(print_vel)
        #training_actions.append(torch.mean(actions.detach(), dim=0))
        # if print_acc < th_error:
        #     break

        ep_rwd = []
        ep_vel = []


#print(s_file)
#torch.save(agent.state_dict(), '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_Reinf_Actor_s35_final_10arms.pt')
# torch.save(training_acc, acc_file)
# torch.save(training_vel, vel_file)

tst_actions = (agent(target_state,True)).view(1, 2, -1)
test_arm = Parall_Arm_model(tspan,x0,dev, n_arms=1)

_, thetas = test_arm.perform_reaching(t_step, tst_actions.detach())



rwd = torch.sqrt(training_arm.compute_rwd(thetas, x_hat, y_hat, f_points))
velocity = torch.sqrt(training_arm.compute_vel(thetas, f_points))

print("tst rwd: ", rwd)
print("tst vel: ",velocity)


