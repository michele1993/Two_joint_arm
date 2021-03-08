import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self,n_arms, Input_size=7, h1_size=56,h2_size=56, Output_size=3,ln_rate = 1e-3):

        super().__init__()

        self.n_arms = n_arms

        self.l1 = nn.Linear(Input_size, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        self.l3 = nn.Linear(h2_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))

        return torch.cat([x[:,0:2], torch.clip(x[:,2:3], 0)],dim=1)


    def freeze_params(self):

        for params in self.parameters():

            params.requires_grad = False

    def update(self,loss):

        loss = torch.mean(loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def copy_weights(self, estimate):

        for t_param, e_param in zip(self.parameters(), estimate.parameters()):
            t_param.data.copy_(e_param.data)

    def soft_update(self, estimate, decay):

        with torch.no_grad():
            # do polyak averaging to update target NN weights
            for t_param, e_param in zip(self.parameters(),estimate.parameters()):
                t_param.data.copy_(e_param.data * decay + (1 - decay) * t_param.data)



class Critic_NN(nn.Module):


    def __init__(self,dev,state_s = 7,a1_s = 50,a2_s = 20,h1_s = 56,h2_s = 56, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        # Initialise mean values for RBF receptive field, based on min/max control signal
        self.mu_s1 = torch.linspace(-2.5,2.5,a1_s).view(1,1,-1).repeat(1,2,1).to(dev) # use this shape for parallelisation, 2 is the size of actions
        #self.sigma1 = 0.06
        self.sigma1 = 0.01

        self.mu_s2 = torch.linspace(0,2.5,a2_s).view(1,1,-1).to(dev)
        #self.sigma2 = 0.075
        self.sigma2 = 0.01

        self.l1 = nn.Linear(state_s + a1_s*2 + a2_s ,h1_s)
        self.l2 = nn.Linear(h1_s,h2_s)
        self.l3 = nn.Linear(h2_s,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        a1 = self.radialBasis_f(a[:,0:2],self.mu_s1,self.sigma1)
        a2 = self.radialBasis_f(a[:,2:],self.mu_s2, self.sigma2)

        x = torch.cat([s, a1,a2], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


    def radialBasis_f(self,x, mu_s, sigma):

        batch_s = x.size()[0]
        rpt_field = torch.exp(-0.5*((x.unsqueeze(2) - mu_s)**2)/sigma)

        return rpt_field.view(batch_s,-1)


    def freeze_params(self):

        for params in self.parameters():

            params.requires_grad = False


    def update(self, target, estimate):

        loss = torch.mean((target - estimate)**2)
        self.optimiser.zero_grad()
        loss.backward() #needed for the actor
        self.optimiser.step()

        return loss

    def copy_weights(self,estimate):

        for t_param, e_param in zip(self.parameters(),estimate.parameters()):
            t_param.data.copy_(e_param.data)

    def soft_update(self, estimate, decay):

        with torch.no_grad():
          # do polyak averaging to update target NN weights
            for t_param, e_param in zip(self.parameters(),estimate.parameters()):
                t_param.data.copy_( e_param.data * decay + (1 - decay) * t_param.data)