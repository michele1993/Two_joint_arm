import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal


# create agent using REINFORCE

class FB_Reinf_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,dev,input_size = 6, n_hiddens = 128,n_outputs = 2, std = 10, ln_rate= 0.1):

        super().__init__()

        self.dev = dev
        self.std = std

        self.l1 = nn.Linear(input_size,n_hiddens)
        self.l2 = nn.Linear(n_hiddens,n_outputs)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)
        self.store_logs = []


    def forward(self,x,train): # sample all control signals in one go and store their log p

        cos_t1 = torch.cos(x[:,0])
        sin_t1 = torch.sin(x[:,0])
        vel_t1 = x[:,2]

        cos_t2 = torch.cos(x[:,1])
        sin_t2 = torch.sin(x[:,1])
        vel_t2 = x[:, 3]

        inpt = torch.cat([cos_t1,sin_t1,vel_t1,cos_t2,sin_t2,vel_t2], dim=1) #CHECK DIM CORRECT! NETWORK takes input batchxsizex1 ?

        inpt = F.relu(self.l1(inpt))
        inpt = self.l2(inpt)

        #inpt = torch.clip(inpt,-20,20)

        if train:
            d = Normal(inpt, self.std)

            inpt = d.sample()
            log_ps = d.log_prob(inpt)

            self.store_logs.append(log_ps)

        return torch.unsqueeze(inpt,dim=2) # need to add third dimension for the dynamical system



    def update(self, rwd):

        log_ps = torch.stack(self.store_logs)
        self.store_logs = []

        loss = torch.sum(log_ps * rwd)

        self.optimiser.zero_grad()
        loss.backward()

        self.optimiser.step()

        self.log_ps = None # flush self.log just in case

        return loss


    # Note: this type of rwd f() may not work for the current problem, the Matlab rwd f() may be a better option
    # due to the rwd being negative and at the final step only and the rest being zero, thus if apply backward discounting
    # initial actions returns will be smaller (thus better) than final actions
    def compute_discounted_returns(self,rwd): # compute the correct discounted rwd

        rwds = torch.Tensor(rwd).repeat(len((self.mu_s)))

        discounts = self.discount ** (torch.FloatTensor(range(len(rwds))))

        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts
        # the first bit flip(cumsum(flip(...))) computes the cumulative sum using the discount from the first state,
        # (e.g. rwd for last state super discounted), so then need to divide by the (extra) discounting applied to
        # each state to get right amount of rwd from that state (this work since cumsum is linear, so can
        # take the discounting out of the sum and applying it at the end for the cumsum for each state (very clever!).


    def forward_dis_return(self,rwd ): # forward discounting, like in matlab the closer an a to rwd the more discounted (since rwd is negative)

        n = torch.Tensor(range(len(self.mu_s))).to(self.dev)

        return rwd * self.discount**n

    def test_actions(self):
        return self.mu_s.T


    def gaussian_convol(self,actions):

        kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061,0.006]]]).to(self.dev)

        actions[:,0:1,:] =  nn.functional.conv1d(actions[:,0:1,:], kernel,padding=(kernel.size()[-1]-1)//2)

        actions[:,1:2,:] =  nn.functional.conv1d(actions[:,1:2,:], kernel,padding=(kernel.size()[-1]-1)//2)

        return actions