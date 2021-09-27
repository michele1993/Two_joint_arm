from Model_based.MB_Arm_model import MB_FF_Arm_model
from Model_based.MB_NN_Agent import MB_Actor_NN
from Model_based.Model_based_alg import MB_alg
import torch
import numpy as np

class ModelBased_train:


    def __init__(self,ln_rate_m,ln_rate_a,episodes, dev):

        self.dev = dev
        self.Overall_episodes = episodes
        self.n_RK_steps = 99
        self.time_window = 0
        self.n_parametrised_steps = self.n_RK_steps -self.time_window
        self.tspan = [0, 0.4]
        self.x0 = [[-np.pi / 2], [np.pi / 2], [0], [0], [0], [0], [0], [0]] # initial condition, needs this shape
        self.t_step = self.tspan[-1]/self.n_RK_steps
        self.f_points = -self.time_window -1
        self.ln_rate_a = ln_rate_a #0.01 works best #works well: 0.001 # 0.00001
        self.velocity_weight = 0.005
        self.max_u = 15000
        self.th_error = 0.01#0.025
        self.n_arms = 1 #10 #100
        self.Model_ln_rate = ln_rate_m #0.1 works best  #works well0.05#0.01 #0.08



        # Target endpoint, based on matlab - reach straight in front, at shoulder height
        self.x_hat = 0.792
        self.y_hat = 0

        self.target_state = torch.tensor([self.x_hat,self.y_hat]).view(1,2).to(self.dev)


        self.target_arm = MB_FF_Arm_model(False,self.tspan,self.x0,self.dev, n_arms=self.n_arms)
        self.estimated_arm = MB_FF_Arm_model(True,self.tspan,self.x0,self.dev, n_arms=self.n_arms,ln_rate = self.Model_ln_rate)

        self.agent = MB_Actor_NN(self.max_u,self.dev,Output_size= self.n_parametrised_steps*2, ln_rate= self.ln_rate_a)
        self.agent.apply(self.agent.small_weight_init)

        self.MB_alg = MB_alg(self.estimated_arm,self.agent ,self.t_step, self.n_parametrised_steps,self.velocity_weight, self.th_error)


    def train(self):

        ep_distance = []
        ep_model_ups = []
        ep_actor_ups = []

        for ep in range(1,self.Overall_episodes):


            actions = self.agent(self.target_state).view(1,2,self.n_parametrised_steps).detach()


            target_ths = self.target_arm.perform_reaching(self.t_step, actions)

            rwd = self.target_arm.compute_rwd(target_ths, self.target_state[0,0], self.target_state[0,1], self.f_points)
            velocity = self.target_arm.compute_vel(target_ths, -1)

            acc = torch.sqrt(rwd)
            ep_distance.append(acc)


            print("Rollouts: ", ep)
            print("Overall Accuracy: ", acc)
            print("Overall Velocity: ", torch.mean(torch.sqrt(velocity)), "\n")


            ep_modelUp = self.MB_alg.update_model(actions.detach(), target_ths, self.target_arm)

            ep_model_ups.append(ep_modelUp)
            print("Eps to update model: ", ep_modelUp)

            ep_actorUp = self.MB_alg.update_actor(self.target_state)
            ep_actor_ups.append(ep_actorUp)


        return ep_distance, ep_model_ups, ep_actor_ups

