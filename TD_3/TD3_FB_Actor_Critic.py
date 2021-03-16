import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self,n_arms, Input_size=7, h1_size=256,h2_size=256, Output_size=3,ln_rate = 1e-3):

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


    def __init__(self,lamb,dev,state_s = 7,a_s = 3,h1_s = 256,h2_s = 256, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        #self.lamb = 1.8
        self.lamb = lamb

        self.l1 = nn.Linear(state_s + a_s ,h1_s)
        self.l2 = nn.Linear(h1_s,h2_s)
        self.l3 = nn.Linear(h2_s,Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        x = self.l1(torch.cat([s, a], dim=1))

        # if torch.sum(torch.isnan(F.relu(x))) > 0:
        #     print('l1')
        #     exit()
        #
        # if torch.sum(torch.isnan(torch.exp(-self.lamb * x))) > 0:
        #     print('exp_l1')
        #     print()
        #     exit()

        #c = x
        x = F.relu(x) * torch.exp(-self.lamb * (x)**2)
        #x = F.relu(x) * torch.exp(-self.lamb * (x))

        # if torch.sum(torch.isnan(x)) > 0:
        #     print(c[torch.isnan(x)])
        #     print(F.relu(c)[torch.isnan(x)])
        #     print(torch.exp(-self.lamb * c)[torch.isnan(x)])
        #     print('p_l2')
        #     exit()

        x = self.l2(x)

        # if torch.sum(torch.isnan(x)) > 0:
        #     print('l2')
        #     exit()
        #
        # if torch.sum(torch.isnan(F.relu(x))) > 0:
        #     print('R_l2')
        #     print(x[torch.isnan(F.relu(x))],'\n')
        #     print(F.relu(x)[torch.isnan(F.relu(x))])
        #     exit()

        x = F.relu(x) * torch.exp(-self.lamb * (x)**2)
        #x = F.relu(x) * torch.exp(-self.lamb * (x))

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