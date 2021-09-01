from MB_DPG.FeedForward.Multi_target.MultiP_learnArm_model import Multi_learnt_ArmModel
from TD_3.FeedForward.FF_AC import *
from TD_3.FeedForward.FF_parall_arm import FF_Parall_Arm_model
import torch
import numpy as np

class MB_DPG_train:

    def __init__(self,model_ln,actor_ln,std,episodes,dev):

        self.dev = dev
        self.episodes = episodes
        self.n_RK_steps = 99
        self.time_window_steps = 0
        self.n_parametrised_steps = self.n_RK_steps - self.time_window_steps
        self.t_print = 100
        self.n_arms = 10 # n. of arms for each target
        self.tspan = [0, 0.4]
        self.x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]]  # initial condition, needs this shape
        self.t_step = self.tspan[-1] / self.n_RK_steps
        self.f_points = -self.time_window_steps -1 # use last point with no zero action # number of final points to average across for distance to target and velocity
        self.vel_weight = 0.05#0.2#0.4
        ln_rate_a = actor_ln#working well: 0.00005
        model_lr = model_ln # working well: 0.001
        self.std = std #working well: 0.01
        self.max_u = 15000
        self.start_a_upd = 500 # 1000 performs much worse
        self.a_size = self.n_parametrised_steps *2
        self.est_y_size = 4 # attempt to predict only 4 necessary components to estimated the rwd (ie. 2 angles and 2 angle vels)
        self.actor_update = 3
        self.std_decay = 0.999# 0.9999 with decay every 100 steps was working good
        self.n_target_p = 50
        self.overall_n_arms = self.n_target_p * self.n_arms # n. of parallel simulations (i.e. n. of amrms x n. of targets)


        self.training_arm = FF_Parall_Arm_model(self.tspan, self.x0, dev, n_arms=self.overall_n_arms)
        self.est_arm = Multi_learnt_ArmModel(output_s = self.est_y_size, ln_rate= model_lr).to(dev)

        # Initialise actor and critic
        self.agent = Actor_NN(dev, Output_size=self.n_parametrised_steps * 2, ln_rate=ln_rate_a).to(dev)
        self.agent.apply(self.agent.small_weight_init)

        # Use to randomly generate targets in front of the arm and on the max distance circumference
        #target_states = training_arm.circof_random_tagrget(n_target_p)
        self.target_states = torch.load('/home/px19783/Two_joint_arm/MB_DPG/FeedForward/Multi_target/Results/MultiPMB_DPG_FF_targetPoints_s1_2.pt')


    def train(self):

        ep_rwd = []
        ep_vel = []



        for ep in range(1, self.episodes):

            det_actions = self.agent(self.target_states)  # may need converting to numpy since it's a tensor

            # add noise to each action for each arm for each target
            exploration = (torch.randn((self.overall_n_arms, 2, self.n_parametrised_steps)) * self.std).to(self.dev)

            # need to repeat the deterministic action for each arm so can add noise to each
            actions = det_actions.repeat(self.n_arms,1).view(self.overall_n_arms, 2, self.n_parametrised_steps) + exploration # need to repeat action means so that can add noise for each target x arm cobination

            simulator_actions = actions * self.max_u

            t, thetas = self.training_arm.perform_reaching(self.t_step, simulator_actions.detach())

            # extract two angles and two angle vel for all arms as target to the estimated model
            thetas = thetas[-1:,:,0:self.est_y_size]#.squeeze(dim=-1) # only squeeze last dim, because if squeeze all then size no longer suitable for compute rwd/vel

            # Update the estimated model:
            est_y = self.est_arm(actions.view(self.overall_n_arms,self.a_size).detach()) # compute y prediction based on current action

            _ = self.est_arm.update(thetas.squeeze(dim=-1), est_y)

            # compute all the necessary gradients for chain-rule update of actor
            diff_thetas = thetas.clone().detach().requires_grad_(True)  # wrap thetas around differentiable tensor to compute dr/dy with autograd.grad


            # Change this (?)
            rwd = self.training_arm.multiP_compute_rwd(diff_thetas,self.target_states[:,0:1],self.target_states[:,1:2], self.f_points, self.n_arms)
            vel = self.training_arm.compute_vel(diff_thetas, self.f_points)

            if ep > self.start_a_upd and ep % self.actor_update ==0:

                # Note: use sum to obtain a scalar value, which can be passed to autograd.grad, it's fine since all the grad for each arm are independent
                weight_rwd = torch.sum(rwd + vel * self.vel_weight)

                # compute gradient of rwd with respect to outcome
                dr_dy = torch.autograd.grad(outputs=weight_rwd, inputs = diff_thetas)[0]


                est_y = self.est_arm(actions.view(self.overall_n_arms, self.a_size)) # re-estimate model prediction since model has been updated and gradient changed

                # compute gradient of rwd with respect to actions, using environment outcome
                dr_da = torch.autograd.grad(outputs= est_y, inputs = actions, grad_outputs= dr_dy.squeeze())[0]

                self.agent.MB_update(actions,dr_da)


            # Estimate for accuracy only
            ep_rwd.append(torch.mean(torch.sqrt(rwd.detach())))
            ep_vel.append(torch.mean(torch.sqrt(vel.detach())))

            # if ep % 100 == 0: # 10 decay std
            #     self.std *= self.std_decay

            if ep % self.t_print == 0:

                print_acc = sum(ep_rwd) / self.t_print
                print_vel = sum(ep_vel) / self.t_print

                ep_rwd = []
                ep_vel = []
                training_acc = print_acc
                training_vel = print_vel

        return training_acc, training_vel