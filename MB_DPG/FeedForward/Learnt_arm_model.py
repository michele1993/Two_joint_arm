import torch
import torch.nn as nn
import torch.optim as op

class learnt_ArmModel(nn.Module):

    def __init__(self, action_s = 198 ,h_s = 118,h_s2=118, output_s = 4,ln_rate=1e-3): #h_s = 400,h_s2=300

        super().__init__()
        self.l1 = nn.Linear(action_s,h_s)
        self.l2 = nn.Linear(h_s,h_s2)
        self.l3 = nn.Linear(h_s2,h_s2)
        self.l4 = nn.Linear(h_s2,output_s)

        self.optimiser = op.Adam(self.parameters(),lr=ln_rate)

    def forward(self, x):

        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        #x = torch.relu(self.l3(x))

        return self.l4(x)

    def update(self, target, estimate):

        loss = torch.mean((estimate - target.squeeze(dim=0))**2) # squeeze only first dim to avoid squeezing dim of arm in case use one arm only
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss


