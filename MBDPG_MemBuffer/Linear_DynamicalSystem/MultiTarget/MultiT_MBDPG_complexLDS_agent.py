import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

class Complex_Linear_DS_agent(nn.Module):

    def __init__(self, t_step,n_steps,ln_rate_1,ln_rate_2, n_targets ,dev,action_s=2,h_s = 10, input_s=2, h_nn_s = 256 ): # h_s = 25; h_nn_s = 116

        super().__init__()

        self.t_step = t_step
        self.n_steps = n_steps
        self.action_s = action_s
        self.h_s = h_s
        self.n_targets = n_targets
        self.dev = dev

        # Define the dynamical system
        self.A = torch.randn((h_s,h_s)).to(self.dev) * 0.1
        self.A.requires_grad = True

        # Define an NN that gives an initial condition based on the target
        self.l1 = nn.Linear(input_s,h_nn_s)
        self.l2 = nn.Linear(h_nn_s,h_s)

        # Define tanh mapping that maps hidden state to action

        self.lm = nn.Linear(h_s,action_s)

        #self.optimiser = opt.Adam(self.parameters(), ln_rate)

        self.optimiser_1 = opt.Adam([self.A], ln_rate_1)
        self.optimiser_2 = opt.Adam(self.parameters(), ln_rate_2)

    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.00005)# std= 0.0005
            nn.init.normal_(l.bias,mean=0,std= 0.00005)# std= 0.0005



    def forward(self, target):

        # compute the initial condition
        x = self.inital_cond(target).squeeze().T

        hidden_states = torch.zeros((self.n_steps,self.h_s,self.n_targets)).to(self.dev)
        actions = torch.zeros((self.n_targets, self.action_s, self.n_steps)).to(self.dev)

        # Peform last mapping out of the loop so can store gradient for the last pair of actions
        for s in range(0, self.n_steps):

            hidden_states[s,:,:] = x.clone().squeeze()
            actions[:,:,s] = self.map_actions(x.T)

            x = x -(self.A@x) * self.t_step


        return actions, hidden_states


    def map_actions(self,x):

        return torch.tanh(self.lm(x))


    def inital_cond(self,x):

        x = F.relu(self.l1(x))

        return self.l2(x)

    def MB_update(self,actions ,gradient):

        actions.backward(gradient=gradient)
        self.optimiser_1.step()
        self.optimiser_2.step()
        self.optimiser_1.zero_grad()
        self.optimiser_2.zero_grad()
