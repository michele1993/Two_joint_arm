import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self, Input_size=2, h1_size=400,h2_size=300, Output_size=170,ln_rate = 1e-3):

        super().__init__()

        self.output_s = Output_size

        self.l1 = nn.Linear(Input_size, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        self.l3 = nn.Linear(h2_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))

        return x

    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std=0.00025)
            nn.init.normal_(l.bias,mean=0,std=.00025)


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


    def __init__(self,input_size = 202,h1_s = 400,h2_s = 300, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        self.input_s = input_size

        self.l1 = nn.Linear(input_size,h1_s)
        self.l2 = nn.Linear(h1_s,h2_s)
        self.l3 = nn.Linear(h2_s,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        x = F.relu(self.l1(torch.cat([s, a], dim=1)))

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

        return loss,

    def copy_weights(self,estimate):

        for t_param, e_param in zip(self.parameters(),estimate.parameters()):
            t_param.data.copy_(e_param.data)

    def soft_update(self, estimate, decay):

        with torch.no_grad():
          # do polyak averaging to update target NN weights
            for t_param, e_param in zip(self.parameters(),estimate.parameters()):
                t_param.data.copy_( e_param.data * decay + (1 - decay) * t_param.data)