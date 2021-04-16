import torch
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Normal


# create agent using REINFORCE

class S_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,n,dev, ln_rate= 50):

        super().__init__()

        self.dev = dev
        action_0 = torch.randn(1,2,n).to(self.dev) *10
        action_0 = self.gaussian_convol(action_0)

        self.actions = nn.Parameter(action_0) # initalise means randomly
        self.decay = nn.Parameter(torch.rand(1)* 10) #

        # Each of dict will define a separate parameter group, and should contain a params key,
        # containing a list of parameters belonging to it. This allows to used separate ln_rate
        self.optimiser1 = opt.Adam([
            {'params': [self.actions]},
            {'params': [self.decay], 'lr': 1e-1} #1e-2
        ],ln_rate)




    def give_parameters(self): # sample all control signals in one go and store their log p

        return self.actions, self.decay  #, torch.clip(self.decay,0,1.73) # 1.73 because it is highest value at which t=0.4, e^(1.73*t) <1



    def update(self, loss):

        self.optimiser1.zero_grad()
        loss.backward()
        self.optimiser1.step()



    def gaussian_convol(self,actions):

        kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061,0.006]]]).to(self.dev)


        actions[:,0:1,:] =  nn.functional.conv1d(actions[:,0:1,:], kernel,padding=(kernel.size()[-1]-1)//2)

        actions[:,1:2,:] =  nn.functional.conv1d(actions[:,1:2,:], kernel, padding=(kernel.size()[-1]-1)//2)


        return actions