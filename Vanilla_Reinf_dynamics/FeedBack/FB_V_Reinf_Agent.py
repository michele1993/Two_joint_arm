import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal


# create agent using REINFORCE

class FB_Reinf_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,dev,n_arms,input_size = 7, n_hiddens = 128,n_outputs = 3, std = 0.5, ln_rate= 0.0005, discount = 0.95):#ln_rate= 0.0005

        super().__init__()

        self.dev = dev
        self.std = std
        self.discount = discount
        self.n_arms = n_arms

        self.l1 = nn.Linear(input_size,n_hiddens)
        self.l2 = nn.Linear(n_hiddens,n_hiddens)
        self.l3 = nn.Linear(n_hiddens,n_outputs)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)
        self.store_logs = []


    def forward(self,x,t,train): # sample all control signals in one go and store their log p

        cos_t1 = torch.cos(x[:,0])
        sin_t1 = torch.sin(x[:,0])
        vel_t1 = x[:,2]

        cos_t2 = torch.cos(x[:,1])
        sin_t2 = torch.sin(x[:,1])
        vel_t2 = x[:, 3]

        inpt = torch.cat([cos_t1,sin_t1,vel_t1,cos_t2,sin_t2,vel_t2,t.expand(self.n_arms,1)], dim=1) #CHECK?! first dim of expand should be n_arms, I belive

        inpt = F.relu(self.l1(inpt))
        inpt = F.relu(self.l2(inpt))
        inpt = self.l3(inpt)


        if train:
            d = Normal(inpt, self.std)
            inpt = d.sample()
            log_ps = d.log_prob(inpt)

            self.store_logs.append(log_ps)

        return torch.unsqueeze(inpt[:,0:2],dim=2), torch.clip(inpt[:,2],0, 200).view(-1,1,1) # need to add third dimension for the dynamical system


    def update(self, rwd):


        log_ps = torch.stack(self.store_logs)

        self.store_logs = []

        loss = torch.sum(log_ps * rwd)

        self.optimiser.zero_grad()
        loss.backward()

        self.optimiser.step()

        return loss


    # Note: this type of rwd f() may not work for the current problem, the Matlab rwd f() may be a better option
    # due to the rwd being negative and at the final step only and the rest being zero, thus if apply backward discounting
    # initial actions returns will be smaller (thus better) than final actions
    def compute_discounted_returns(self,rwd,steps): # compute the correct discounted rwd


        rwds = torch.Tensor(rwd).repeat(steps).to(self.dev)

        discounts = self.discount ** (torch.FloatTensor(range(len(rwds))).to(self.dev))

        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts
        # the first bit flip(cumsum(flip(...))) computes the cumulative sum using the discount from the first state,
        # (e.g. rwd for last state super discounted), so then need to divide by the (extra) discounting applied to
        # each state to get right amount of rwd from that state (this work since cumsum is linear, so can
        # take the discounting out of the sum and applying it at the end for the cumsum for each state (very clever!).


    def forward_dis_return(self,rwd,steps ): # forward discounting, like in matlab the closer an a to rwd the more discounted (since rwd is negative)

        n = torch.Tensor(range(steps)).to(self.dev)

        return rwd * self.discount**n

    def test_actions(self):
        return self.mu_s.T


    def gaussian_convol(self,actions):

        kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061,0.006]]]).to(self.dev)

        actions[:,0:1,:] =  nn.functional.conv1d(actions[:,0:1,:], kernel,padding=(kernel.size()[-1]-1)//2)

        actions[:,1:2,:] =  nn.functional.conv1d(actions[:,1:2,:], kernel,padding=(kernel.size()[-1]-1)//2)

        return actions