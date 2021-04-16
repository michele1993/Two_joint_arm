import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal


class Reinf_Actor_NN(nn.Module):

    def __init__(self,std,n_arms,max_u,dev, Input_size=2, h1_size=256,h2_size=256,h3_size=256, Output_size=198,ln_rate = 1e-3):

        super().__init__()

        self.output_s = Output_size
        self.std = std
        self.dev = dev
        self.n_arms = n_arms
        self.max_u = max_u

        self.l1 = nn.Linear(Input_size, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        self.l3 = nn.Linear(h2_size, h3_size)
        self.l4 = nn.Linear(h3_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x,test):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = torch.tanh(self.l4(x))

        # check x dim correct for distribution
        if not test:
            d = Normal(x, self.std)

            sampled_as = d.sample((self.n_arms,))
            self.log_ps = d.log_prob(sampled_as).squeeze()


            return sampled_as.view(self.n_arms,2,-1) * self.max_u

        else:

            return x * self.max_u


    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.00005)# std= 0.00005
            nn.init.normal_(l.bias,mean=0,std= 0.00005)# std= 0.00005



    def update(self,dis_rwd):

        dis_rwd = torch.sum(dis_rwd,dim=0)
        loss = torch.sum(self.log_ps * dis_rwd)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.log_ps = None # flush self.log just in case

        return loss

