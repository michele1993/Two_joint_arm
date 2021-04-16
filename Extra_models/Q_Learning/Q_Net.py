import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Critic_NN(nn.Module):


    def __init__(self,state_s = 6,action_s =3,hidden_st = 128,hidden_a = 64, Hidden_size = 256, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        self.l0s = nn.Linear(state_s,hidden_st)
        self.l0a = nn.Linear(action_s,hidden_a)

        self.l1 = nn.Linear(hidden_st + hidden_a,Hidden_size)
        self.l2 = nn.Linear(Hidden_size,Output_size)


        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        # Process actions and states separately for first layer only
        s = self.compute_state(s)
        s = F.relu(self.l0s(s))
        a = F.relu(self.l0a(a))

        x = torch.cat([s,a],dim=1)
        x = F.relu(self.l1(x))
        x = self.l2(x)

        return x

    def compute_state(self, s):

        cos_t1 = torch.cos(s[:, 0])
        sin_t1 = torch.sin(s[:, 0])
        vel_t1 = s[:, 2]

        cos_t2 = torch.cos(s[:, 1])
        sin_t2 = torch.sin(s[:, 1])
        vel_t2 = s[:, 3]

        return torch.cat([cos_t1, sin_t1, vel_t1, cos_t2, sin_t2, vel_t2], dim=1)


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