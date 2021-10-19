import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal

class Linear_DS_agent(nn.Module):

    def __init__(self, std,max_u,t_step,n_steps,ln_rate ,n_arms,dev,action_s=2,h_s = 10, input_s=2, h_nn_s = 256):

        super().__init__()

        self.std = std
        self.max_u = max_u
        self.t_step = t_step
        self.n_steps = n_steps
        self.action_s = action_s
        self.h_s = h_s
        self.dev = dev
        self.n_arms = n_arms
        self.output_s = n_steps *2

        # Define the dynamical system
        self.D = torch.randn(h_s).to(self.dev) * 0.1#0.001 # initialise a random diagonal matrix
        self.P = torch.randn((h_s,h_s)).to(self.dev)

        self.D = nn.Parameter(self.D)
        self.P = nn.Parameter(self.P)

        # Define an NN that gives an initial condition based on the target
        self.l1 = nn.Linear(input_s,h_nn_s)
        self.l2 = nn.Linear(h_nn_s,h_s)

        # Define tanh mapping that maps hidden state to action

        self.lm = nn.Linear(h_s,action_s)

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.00005)# std= 0.0005
            nn.init.normal_(l.bias,mean=0,std= 0.00005)# std= 0.0005

    def compute_A(self):

        return self.P @ torch.diag_embed(torch.exp(self.D)) @ torch.inverse(self.P)



    def forward(self, target, test = False):

        # compute the initial condition
        x = self.inital_cond(target).squeeze()
        A = self.compute_A()

        hidden_states = torch.zeros((self.n_steps,self.h_s)).to(self.dev)

        # Peform last mapping out of the loop so can store gradient for the last pair of actions
        for s in range(0, self.n_steps):

            hidden_states[s,:] = x.clone().squeeze()
            x = x -(A@x) * self.t_step


        actions = self.map_actions(hidden_states)


        if not test:
            d = Normal(actions, self.std) # build Gaussian for each target, , if multiples used

            sampled_as = d.sample((self.n_arms,)) # take a sample for each arm


            self.log_ps = d.log_prob(sampled_as).view(-1, self.output_s)


            return sampled_as.view(self.n_arms,2,self.n_steps) * self.max_u

        else:

            return x * self.max_u


    def map_actions(self,x):

        return torch.tanh(self.lm(x))


    def inital_cond(self,x):

        x = F.relu(self.l1(x))

        return self.l2(x)

    def update(self,dis_rwd):

        dis_rwd = torch.sum(dis_rwd,dim=0) # needed in case include multiple final points

        loss = torch.sum(self.log_ps * dis_rwd)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.log_ps = None # flush self.log just in case

        return loss






