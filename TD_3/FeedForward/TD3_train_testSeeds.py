from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np



class TD3_train:

    def __init__(self, critic_ln,actor_ln,std,episodes,dev):

        self.dev = dev
        self.episodes = episodes
        self.n_RK_steps = 99
        self.time_window_steps = 0
        self.n_parametrised_steps = self.n_RK_steps - self.time_window_steps
        self.t_print = 100
        self.n_arms = 10 #100
        self.tspan = [0, 0.4]
        self.x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
        self.t_step = self.tspan[-1] / self.n_RK_steps
        self.f_points = -self.time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
        self.vel_weight = 0.005
        self.ln_rate_c = actor_ln
        self.ln_rate_a = critic_ln
        self.std = std
        self.max_u = 15000
        self.start_a_upd = 100
        self.th_conf = 0.85

        # Target endpoint, based on matlab - reach straight in front, at shoulder height
        self.x_hat = 0.792
        self.y_hat = 0

        self.target_state = torch.tensor([self.x_hat, self.y_hat]).view(1, 2).to(dev)  # .repeat(n_arms,1).to(dev)


        self.training_arm = FF_Parall_Arm_model(self.tspan, self.x0, dev, n_arms=self.n_arms)

        # Initialise actor and critic
        self.agent = Actor_NN(dev, Output_size=self.n_parametrised_steps * 2, ln_rate=self.ln_rate_a).to(dev)
        self.agent.apply(self.agent.small_weight_init)

        self.c_input_s = self.n_parametrised_steps * 2 + 2
        self.critic_1 = Critic_NN(self.n_arms, dev, input_size=self.c_input_s, ln_rate=self.ln_rate_c).to(dev)


    def train(self):

        ep_rwd = []
        training_acc = []


        for ep in range(1, self.episodes):

            det_actions = self.agent(self.target_state)  # may need converting to numpy since it's a tensor

            exploration = (torch.randn((self.n_arms, 2, self.n_parametrised_steps)) * self.std).to(self.dev)

            Q_actions = (det_actions.view(1, 2, -1) + exploration).detach()

            actions = Q_actions * self.max_u

            t, thetas = self.training_arm.perform_reaching(self.t_step, actions.detach())

            rwd = self.training_arm.compute_rwd(thetas, self.x_hat, self.y_hat, self.f_points)
            velocity = self.training_arm.compute_vel(thetas, self.f_points)

            acc_rwd = rwd.clone()


            weighted_adv = (rwd + self.vel_weight * velocity)

            Q_v = self.critic_1(self.target_state, Q_actions.view(self.n_arms, -1), False)
            _ = self.critic_1.update(weighted_adv, Q_v)

            Tar_Q = self.critic_1(self.target_state, det_actions, True)  # want to max the advantage

            # UNCOMMENT for multiple arms, best version
            mean_G = torch.mean(torch.mean(weighted_adv, dim=0), dim=0)
            std_G = torch.sqrt(torch.sum((mean_G - weighted_adv) ** 2) / (self.n_arms - 1))
            confidence = torch.abs((Tar_Q - mean_G) / std_G).detach()

            if ep > self.start_a_upd and confidence <= self.th_conf:

                self.agent.update(Tar_Q)

            ep_rwd.append(torch.mean(torch.sqrt(acc_rwd)))

            if ep % self.t_print == 0:

                print_acc = sum(ep_rwd) / self.t_print

                ep_rwd = []
                training_acc.append(print_acc)


        return training_acc