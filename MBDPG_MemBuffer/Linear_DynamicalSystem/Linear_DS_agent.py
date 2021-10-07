import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

class Linear_DS_agent(nn.Module):

    def __init__(self, n_steps,action_s=2,h_s = 10, input_s=2, h_nn_s = 116):

        super().__init__()

        self.n_steps = n_steps
        self.action_s = action_s
        self.h_s = h_s

        # Define the dynamical system
        self.D = torch.diag(torch.rand(h_s)) # initialise a random diagonal matrix
        self.P = torch.randn((h_s,h_s))

        self.D = nn.Parameter(self.D)
        self.P = nn.Parameter(self.P)

        # Define an NN that gives an initial condition based on the target
        self.l1 = nn.Linear(input_s,h_nn_s)
        self.l2 = nn.Linear(h_nn_s,h_s)

        # Define tanh mapping that maps hidden state to action

        self.lm = nn.Linear(h_nn_s,action_s)

    def compute_A(self):

        return self.P @ self.D @ torch.inverse(self.P)

    def forward(self, target):

        # compute the initial condition
        x = self.inital_cond(target)
        A = self.compute_A()

        actions = torch.zeros((self.n_steps * self.action_s,1))
        hidden_states = torch.zeros((self.n_steps,self.h_s))

        # Peform last mapping out of the loop so can store gradient for the last pair of actions
        for s in range(0, self.n_steps -1):

            hidden_states[s,:] = x.detach().clone()
            a = self.map_actions(x)
            actions[s:s+1] = a.detach().clone()
            x = A@x

        actions[-2:-1] = self.map_actions(x)
        hidden_states[-1,:] = x.detach().clone()

        return actions, hidden_states


    def MB_update(self,actions ,gradient):

        actions.backward(gradient=gradient)
        self.optimiser.step()
        self.optimiser.zero_grad()


    def map_actions(self,x):

        return F.tanh(self.lm(x))


    def inital_cond(self,x):

        x = F.relu(self.l1(x))

        return self.l2(x)



