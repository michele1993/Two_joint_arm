from Vanilla_Reinf_dynamics.FeedForward.FF_Parall_Arm_model import Parall_Arm_model
from Vanilla_Reinf_dynamics.FeedForward.NN_VanReinf_Agent import Reinf_Actor_NN
import torch
#from safety_checks.Video_arm_config import Video_arm
import numpy as np


class Reinf_train:

    def __init__(self,std,actor_ln,episodes,n_arms,dev):


        self.episodes = episodes
        self.n_RK_steps = 99
        self.time_window_steps = 0
        self.n_parametrised_steps = self.n_RK_steps - self.time_window_steps
        self.t_print = 100
        self.n_arms = n_arms#50#100
        self.tspan = [0, 0.4]
        self.x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
        self.t_step = self.tspan[-1]/self.n_RK_steps
        self.f_points = - self.time_window_steps -1
        self.vel_weight = 0.005
        self.ln_rate = actor_ln
        self.std = std
        self.max_u = 15000
        self.std_decay = 0.999


        # Target endpoint, based on matlab - reach straight in front, at shoulder height
        self.x_hat = 0.792
        self.y_hat = 0
        self.target_state = torch.tensor([self.x_hat,self.y_hat]).view(1,2).to(dev)


        self.training_arm = Parall_Arm_model(self.tspan,self.x0,dev, n_arms=self.n_arms)
        self.agent = Reinf_Actor_NN(std, self.n_arms,self.max_u,dev, ln_rate= self.ln_rate,Output_size=self.n_parametrised_steps*2).to(dev)
        self.agent.apply(self.agent.small_weight_init)


    def train(self):

        ep_rwd = []
        ep_vel = []

        for ep in range(1,self.episodes):

            actions = self.agent(self.target_state,False) # may need converting to numpy since it's a tensor

            t, thetas = self.training_arm.perform_reaching(self.t_step,actions)

            rwd = self.training_arm.compute_rwd(thetas,self.x_hat,self.y_hat, self.f_points)
            velocity = self.training_arm.compute_vel(thetas, self.f_points)

            weighted_adv = rwd + self.vel_weight * velocity

            self.agent.update(weighted_adv)

            ep_rwd.append(torch.mean(torch.sqrt(rwd)))
            ep_vel.append(torch.mean(torch.sqrt(velocity)))

            if ep % 10 == 0:  # decays works better if applied every 10 eps
                self.std *= self.std_decay

            if ep % self.t_print == 0:

                print_acc = sum(ep_rwd)/self.t_print
                print_vel = sum(ep_vel)/self.t_print

                training_acc = print_acc
                training_vel = print_vel

                ep_rwd = []
                ep_vel = []


        return training_acc, training_vel

