import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal


class Reinf_Actor_NN(nn.Module):

    def __init__(self,std,n_arms,max_u,dev, Input_size=2, h1_size=256,h2_size=256,h3_size=256, Output_size=198,ln_rate = 1e-3):

        super().__init__()

        self.output_s = Output_size
        self.std = torch.Tensor([std]).to(dev)
        self.dev = dev
        self.n_arms = n_arms
        self.max_u = max_u
        self.ln_rate = ln_rate

        self.n_action = int(self.output_s / 2) # n of action for each torque

        self.l1 = nn.Linear(Input_size, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        self.l3 = nn.Linear(h2_size, h3_size)
        self.l4 = nn.Linear(h3_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),self.ln_rate)


    def forward(self, x,test):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = torch.tanh(self.l4(x)) # output mean value for each target, if multiples used


        # check x dim correct for distribution
        if not test:
            d = Normal(x, self.std) # build Gaussian for each target, , if multiples used

            sampled_as = d.sample((self.n_arms,)) # take a sample for each arm

            self.log_ps = d.log_prob(sampled_as).view(-1, self.output_s)

            #sampled_as = torch.transpose(sampled_as,0,1) # reshape to targets x n_arms x actions - avoid using it so that can use view after

            return sampled_as.view(-1,2,self.n_action) * self.max_u

        else:

            return x * self.max_u


    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.00005)# std= 0.00005
            nn.init.normal_(l.bias,mean=0,std= 0.00005)# std= 0.00005



    def update(self,dis_rwd):


        dis_rwd = torch.sum(dis_rwd,dim=0) # needed in case include multiple final points
        loss = torch.sum(self.log_ps * dis_rwd)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.log_ps = None # flush self.log just in case

        return loss

    def DPG_update(self, loss):

        loss = torch.mean(loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss


class Critic_NN(nn.Module):


    def __init__(self,n_arms,dev,a_size = 198,s_size=2,h1_s = 116,h1_a= 400,h2_s = 300, Output_size = 1,ln_rate = 0.005):

        super().__init__()


        self.n_arms = n_arms
        self.dev = dev

        self.l1_s = nn.Linear(s_size,h1_s)
        self.l1_a = nn.Linear(a_size, h1_a)
        self.l2 = nn.Linear(h1_s+h1_a,h2_s)
        self.l3 = nn.Linear(h2_s,h2_s)
        self.l4 = nn.Linear(h2_s,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a, det):


        if not det: # for deterministic actions only use 1 arm, since all the same
            s = s.repeat(self.n_arms,1).to(self.dev)

        #x = F.relu(self.l1(torch.cat([s, a], dim=1)))

        s = self.l1_s(s)
        a = self.l1_a(a)

        x = F.relu(self.l2(torch.cat([s, a], dim=1)))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x


    def update(self, target, estimate):

        target = torch.mean(target,dim=0) # sum across time window
        loss = torch.mean((target - estimate)**2)
        self.optimiser.zero_grad()
        loss.backward() #needed for the actor
        self.optimiser.step()

        return loss

