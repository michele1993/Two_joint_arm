from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import Reinf_Actor_NN
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np

# Use REINFORCE to control arm reaches in a feedforward fashion


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dev = torch.device('cpu')


episodes = 1001
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
range_ln_rate = torch.linspace(0.00001,0.001,10)#0.00001
range_std = torch.linspace(0.001,0.05,10)#0.01
max_u = 15000
th_error = 0.025


# Target endpoint, based on matlab - reach straight in front, at shoulder height
x_hat = 0.792
y_hat = 0
target_state = torch.tensor([x_hat,y_hat]).view(1,2).to(dev)

total_acc = torch.zeros(len(range_ln_rate)*len(range_std), 3)
total_vel = torch.zeros(len(range_ln_rate)*len(range_std), 3)

i = 0

for ln_rate in range_ln_rate:

    for std in range_std:

        torch.manual_seed(1)  # FIX SEED

        training_acc = None
        training_vel = None

        training_arm = Parall_Arm_model(tspan,x0,dev, n_arms=n_arms)
        agent = Reinf_Actor_NN(std, n_arms,max_u,dev, ln_rate= ln_rate,Output_size=n_parametrised_steps*2).to(dev)
        agent.apply(agent.small_weight_init)

        ep_rwd = []
        ep_vel = []

        for ep in range(1,episodes):


            actions = agent(target_state,False) # may need converting to numpy since it's a tensor

            t, thetas = training_arm.perform_reaching(t_step,actions)

            rwd = training_arm.compute_rwd(thetas,x_hat,y_hat, f_points)
            velocity = training_arm.compute_vel(thetas, f_points)

            weighted_adv = rwd + vel_weight * velocity

            agent.update(weighted_adv)

            ep_rwd.append(torch.mean(torch.sqrt(rwd)))
            ep_vel.append(torch.mean(torch.sqrt(velocity)))


            if ep % t_print == 0:

                print_acc = sum(ep_rwd)/t_print
                print_vel = sum(ep_vel)/t_print


                print("episode: ", ep, "std: ",std, "ln_rate:", ln_rate)
                print("training accuracy: ",print_acc)
                print("training velocity: ", print_vel,"\n")

                training_acc = print_acc
                training_vel = print_vel
                ep_rwd = []
                ep_vel = []


        total_acc[i,:] = torch.tensor([training_acc, std, ln_rate])
        total_vel[i,:] = torch.tensor([training_vel, std, ln_rate])
        i +=1
        print("iteration n: ", i)


torch.save(total_acc, '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_Reinf_HyperParameter_accur_s1.pt')
torch.save(total_vel, '/home/px19783/Two_joint_arm/Vanilla_Reinf_dynamics/FeedForward/Results/NN_FF_Reinf_HyperParameter_vel_s1.pt')