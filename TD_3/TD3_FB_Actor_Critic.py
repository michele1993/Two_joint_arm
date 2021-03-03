import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self,n_arms, Input_size=7, h1_size=400,h2_size=300, Output_size=3,ln_rate = 1e-3):

        super().__init__()

        self.n_arms = n_arms

        self.l1 = nn.Linear(Input_size, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        #self.l3 = nn.Linear(Hidden_size, Hidden_size)
        self.l4 = nn.Linear(h2_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #x = F.relu(self.l3(x))
        x = torch.tanh(self.l4(x))

        return torch.cat([x[:,0:2] * 2500, torch.clip(x[:,2:3], 0) *200],dim=1)


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


    def __init__(self,dev,state_s = 7,a1_s = 51,a2_s = 1,h1_s = 400,h2_s = 300,hidden_a1 = 118,hidden_a2 = 56, hidden_st = 118, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        # Initialise mean values for RBF receptive field, based on min/max control signal
        self.mu_s = torch.linspace(-2500,2500,a1_s).view(1,1,-1).repeat(1,2,1).to(dev) # use this shape for parallelisation, 2 is the size of actions
        self.sigma = 60

        self.l0s = nn.Linear(state_s,hidden_st)
        self.l0a1 = nn.Linear(a1_s*2,hidden_a1)
        self.l0a2 = nn.Linear(a2_s, hidden_a2)

        self.l1 = nn.Linear(hidden_st + hidden_a1 + hidden_a2 ,h1_s)
        self.l2 = nn.Linear(h1_s,h2_s)
        self.l3 = nn.Linear(h2_s,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        # Process actions and states separately for first layer only
        a1 = self.radialBasis_f(a[:,0:2]) # ,a2
        #a1 = self.radialBasis_f(a)

        s = torch.relu(self.l0s(s))
        a1 = torch.relu(self.l0a1(a1))
        a2 = torch.sigmoid(self.l0a2(a[:,2:])) # use a sigmoid for decay_rate since doesn't use receptive field for it

        x = torch.cat([s,a1,a2],dim=1)
        #x = torch.cat([s, a1], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


    def radialBasis_f(self,a):

        x = a[:,0:2].unsqueeze(2)
        batch_s = a.size()[0]
        rpt_field = torch.exp(-0.5*((x - self.mu_s)**2)/self.sigma)

        return rpt_field.view(batch_s,-1) #, a[:,2:3]


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