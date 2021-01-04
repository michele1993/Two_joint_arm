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


        self.optimiser = opt.Adam(self.parameters(),ln_rate)




    def give_actions(self): # sample all control signals in one go and store their log p

        return self.actions



    def update(self, loss):

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()



    def gaussian_convol(self,actions):

        kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061,0.006]]]).to(self.dev)


        actions[:,0:1,:] =  nn.functional.conv1d(actions[:,0:1,:], kernel,padding=(kernel.size()[-1]-1)//2)

        actions[:,1:2,:] =  nn.functional.conv1d(actions[:,1:2,:], kernel, padding=(kernel.size()[-1]-1)//2)


        return actions