import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt



# create agent for a Supervised FeedBack model

class Spvsd_FB_Parallel_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,n_models,dev,input_size = 7, n_hiddens = 128,n_outputs = 3, ln_rate= 0.05):

        super().__init__()

        self.dev = dev


        self.l1 = nn.Parameter(torch.randn((n_models,n_hiddens,input_size)))
        self.l2 = nn.Parameter(torch.randn((n_models, n_hiddens, n_hiddens)))
        self.l3 = nn.Parameter(torch.randn((n_models, n_outputs, n_hiddens)))

        self.b1 = nn.Parameter(torch.randn((n_models,n_hiddens,1)))
        self.b2 = nn.Parameter(torch.randn((n_models, n_hiddens, 1)))
        self.b3 = nn.Parameter(torch.randn((n_models, n_outputs, 1)))

        self.n_models = n_models

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



    def forward(self,x,t): # sample all control signals in one go and store their log p

        cos_t1 = torch.cos(x[:,0:1])
        sin_t1 = torch.sin(x[:,0:1])
        vel_t1 = x[:,2:3]

        cos_t2 = torch.cos(x[:,1:2])
        sin_t2 = torch.sin(x[:,1:2])
        vel_t2 = x[:, 3:4]


        inpt = torch.cat([cos_t1,sin_t1,vel_t1,cos_t2,sin_t2,vel_t2,t.expand(self.n_models,1,1).to(self.dev)], dim=1) # first dim of expand should be n_arms, I belive

        inpt = F.relu(self.l1 @ inpt + self.b1)
        inpt = F.relu(self.l2 @ inpt + self.b2)
        inpt = self.l3 @ inpt + self.b3

        # need to add third dimension to control signal for the dynamical system
        return inpt[:, 0:2], torch.clip(inpt[:, 2], 0, 200).view(-1, 1, 1)
        #return torch.unsqueeze(inpt[:,0:2],dim=2), torch.clip(inpt[:,2],0, 200) # clip time decay value to avoid "breakin" simulator by flipping the sign


    def update(self, loss):

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss