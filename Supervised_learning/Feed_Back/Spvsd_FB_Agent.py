import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt



# create agent for a Supervised FeedBack model

class Spvsd_FB_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,dev,input_size = 6, n_hiddens = 128,n_outputs = 2, ln_rate= 0.05):

        super().__init__()

        self.dev = dev

        self.l1 = nn.Linear(input_size,n_hiddens)
        self.l2 = nn.Linear(n_hiddens,n_outputs)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



    def forward(self,x): # sample all control signals in one go and store their log p


        cos_t1 = torch.cos(x[:,0])
        sin_t1 = torch.sin(x[:,0])
        vel_t1 = x[:,2]

        cos_t2 = torch.cos(x[:,1])
        sin_t2 = torch.sin(x[:,1])
        vel_t2 = x[:, 3]

        inpt = torch.cat([cos_t1,sin_t1,vel_t1,cos_t2,sin_t2,vel_t2], dim=1) #CHECK DIM CORRECT! NETWORK takes input batchxsizex1 ?

        inpt = F.relu(self.l1(inpt))
        inpt = self.l2(inpt)


        return torch.unsqueeze(inpt,dim=2) # need to add third dimension for the dynamical system


    def update(self, loss):

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss