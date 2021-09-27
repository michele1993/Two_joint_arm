from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
from TD_3.FeedForward.FF_AC import *
import torch
import numpy as np
from MB_DPG.FeedForward.Learnt_arm_model import learnt_ArmModel
from MBDPG_MemBuffer.Memory_Buffer import MemBuffer

class Mbuffer_MBDPG_train:

    def __init__(self,model_ln,actor_ln,std,episodes,n_arms,dev):

        self.dev = dev
        self.episodes = episodes
        self.n_RK_steps = 99
        self.time_window_steps = 0
        self.n_parametrised_steps = self.n_RK_steps - self.time_window_steps
        self.t_print = 100
        self.n_arms = n_arms  # 10
        self.tspan = [0, 0.4]
        self.x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
        self.t_step = self.tspan[-1] / self.n_RK_steps
        self.f_points = -self.time_window_steps - 1  # use last point with no zero action # number of final points to average across for distance to target and velocity
        self.vel_weight = 0.05  # 0.2#0.4
        self.start_a_upd = 100 #500 #10 # 1000 performs much worse
        self.a_size = self.n_parametrised_steps * 2
        self.est_y_size = 4  # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
        self.std_decay = 0.99
        self.max_u = 15000
        self.std = std
        self.model_batch_s = 100 #150
        buffer_size = 500 #5000

        # Target endpoint, based on matlab - reach straight in front, at shoulder height
        self.x_hat = 0.792
        self.y_hat = 0
        self.target_state = torch.tensor([self.x_hat, self.y_hat]).view(1, 2).to(self.dev)



        self.training_arm = FF_Parall_Arm_model(self.tspan, self.x0, self.dev, n_arms=self.n_arms)
        self.est_arm = learnt_ArmModel(output_s=self.est_y_size, ln_rate=model_ln).to(self.dev)

        # Initialise actor and critic
        self.agent = Actor_NN(self.dev, Output_size=self.n_parametrised_steps * 2, ln_rate=actor_ln).to(self.dev)
        self.agent.apply(self.agent.small_weight_init)

        self.M_buffer = MemBuffer(self.n_arms, self.a_size, self.est_y_size,dev, size=buffer_size)

    def train(self):

        ep_rwd = []
        training_acc = []

        std = self.std

        for ep in range(1, self.episodes):

            det_actions = self.agent(self.target_state)  # may need converting to numpy since it's a tensor

            exploration = (torch.randn((self.n_arms, 2, self.n_parametrised_steps)) * std).to(self.dev)

            actions = (det_actions.view(1, 2,  -1) + exploration)  # * max_u , don't apply actual value for action fed to meta model

            simulator_actions = actions * self.max_u

            _, thetas = self.training_arm.perform_reaching(self.t_step, simulator_actions.detach())

            # extract two angles and two angle vel for all arms as target to the estimated model
            thetas = thetas[-1:, :, 0:self.est_y_size].squeeze(dim=-1)  # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

            # ---- compute all the necessary gradients for chain-rule to update the actor ----

            diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy

            rwd = self.training_arm.compute_rwd(diff_thetas, self.x_hat, self.y_hat, self.f_points)
            vel = self.training_arm.compute_vel(diff_thetas, self.f_points)

            weight_rwd = torch.sum(rwd + vel * self.vel_weight)

            # Store transition in the buffer
            self.M_buffer.store(actions.detach().view(self.n_arms, self.a_size), thetas)

            # Sampled from the buffer

            sampled_a, sampled_thetas = self.M_buffer.sample(self.model_batch_s)

            # ---- Update the model based on batch of sampled transitions -------

            est_y = self.est_arm(sampled_a)  # compute y prediction based on current action

            self.est_arm.update(sampled_thetas, est_y)

            # ---- Update the actor based on the actual observed transition -------

            if ep > self.start_a_upd:  # and ep % actor_update ==0:

                # Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent

                # compute gradient of rwd with respect to outcome
                dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs=diff_thetas)[0]

                est_y = self.est_arm(actions.view(self.n_arms, self.a_size))  # re-estimate values since model has been updated

                # compute gradient of rwd with respect to actions, using environment outcome
                dr_da = torch.autograd.grad(outputs=est_y, inputs=actions, grad_outputs=dr_dy.squeeze(0))[0]

                self.agent.MB_update(actions, dr_da)

            # Estimate for accuracy only
            ep_rwd.append(torch.mean(torch.sqrt(rwd.detach())))

            if ep % 10 == 0:  # decays works better if applied every 10 eps
                std *= self.std_decay

            if ep % self.t_print == 0:

                print_acc = sum(ep_rwd) / self.t_print
                ep_rwd = []
                training_acc.append(print_acc)


        return training_acc