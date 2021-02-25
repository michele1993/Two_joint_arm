import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self, Input_size=6, Hidden_size=128, Output_size=3,ln_rate = 1e-3):

        super().__init__()

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size, Hidden_size)
        #self.l3 = nn.Linear(Hidden_size, Hidden_size)
        self.l4 = nn.Linear(Hidden_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x):

        x = self.compute_state(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #x = F.relu(self.l3(x))
        x = self.l4(x)

        return torch.cat([x[:,0:2], torch.clip(x[:,2:3], 0,200)],dim=1)

    def compute_state(self,x):

        cos_t1 = torch.cos(x[:,0])
        sin_t1 = torch.sin(x[:,0])
        vel_t1 = x[:,2]

        cos_t2 = torch.cos(x[:,1])
        sin_t2 = torch.sin(x[:,1])
        vel_t2 = x[:, 3]

        return torch.cat([cos_t1,sin_t1,vel_t1,cos_t2,sin_t2,vel_t2], dim=1)


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


    def __init__(self,dev,state_s = 6,a1_s =5001,a2_s =1,hidden_st = 256,hidden_a = 128, Hidden_size = 256, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        # Initialise mean values for RBF receptive field, based on min/max control signal
        self.mu_s = torch.linspace(-5000,5000,a1_s).view(1,1,-1).repeat(1,2,1).to(dev) # use this shape for parallelisation, 2 is the size of actions
        self.sigma = 5

        self.l0s = nn.Linear(state_s,hidden_st)
        self.l0a1 = nn.Linear(a1_s*2,hidden_st)
        self.l0a2 = nn.Linear(a2_s, hidden_a)

        self.l1 = nn.Linear(hidden_st *2 + hidden_a,Hidden_size)
        #self.l2 = nn.Linear(Hidden_size,Hidden_size)
        self.l3 = nn.Linear(Hidden_size,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        # Process actions and states separately for first layer only
        s = self.compute_state(s)
        a1,a2 = self.radialBasis_f(a)

        s = torch.relu(self.l0s(s))
        a1 = torch.relu(self.l0a1(a1))
        a2 = torch.relu(self.l0a2(a2))
        #a = torch.sigmoid(self.l0a(a))

        x = torch.cat([s,a1,a2],dim=1)
        x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

    def compute_state(self, s):

        cos_t1 = torch.cos(s[:, 0])
        sin_t1 = torch.sin(s[:, 0])
        vel_t1 = s[:, 2]

        cos_t2 = torch.cos(s[:, 1])
        sin_t2 = torch.sin(s[:, 1])
        vel_t2 = s[:, 3]

        return torch.cat([cos_t1, sin_t1, vel_t1, cos_t2, sin_t2, vel_t2], dim=1)

    def radialBasis_f(self,a):

        x = a[:,0:2].unsqueeze(2)
        batch_s = a.size()[0]
        rpt_field = torch.exp(-0.5*((x - self.mu_s)**2)/self.sigma)

        return rpt_field.view(batch_s,-1), a[:,2:3]


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