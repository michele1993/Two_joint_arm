import random
import torch
import numpy as np


class V_Memory_B:

    def __init__(self,n_arms,dev,a_space= 3,s_space = 8,batch_size = 64,size = 20000):

        self.size = size
        self.batch_size = batch_size
        self.s_space = s_space
        self.a_space = a_space

        self.c_size = 0
        self.dev = dev

        # Intialise Tensor buffer for each compotent on CPU to avoid running out of memory
        self.c_state_buf = torch.zeros(self.size,n_arms,s_space,1).to(self.dev)
        self.actions_buf = torch.zeros(self.size,n_arms,a_space).to(self.dev)
        self.rwd_buf = torch.zeros(self.size,n_arms,1).to(self.dev)
        self.n_state_buf = torch.zeros(self.size,n_arms,s_space,1).to(self.dev)
        self.done_buf = torch.zeros(self.size,n_arms).to(self.dev)



    def store_transition(self,c_state,action,rwd,n_state,dn):

        # Determine indx for buffer, based on ratio n of stored steps vs size
        c_idx = self.c_size % self.size # this is 0 whenever buffer completed

        self.c_state_buf[c_idx,:] = c_state.to(self.dev)
        self.actions_buf[c_idx,:] = action.to(self.dev)
        self.rwd_buf[c_idx,:] = rwd.to(self.dev)
        self.n_state_buf[c_idx,:] = n_state.to(self.dev)
        self.done_buf[c_idx,:] = dn.to(self.dev)

        self.c_size+=1



    def sample_transition(self):

        indx_upper_b = min(self.c_size,self.size) # check if buffer is full and take appropriate indx
        indx = torch.randint(0,indx_upper_b,(self.batch_size,)).to(self.dev)

        # Sample corresponding transition
        spl_c_state = self.c_state_buf[indx,:].view(-1,self.s_space,1)#.to('cuda:0')
        spl_a = self.actions_buf[indx,:].view(-1,self.a_space)#.to('cuda:0')
        spl_rwd = self.rwd_buf[indx,:].view(-1,1)#.to('cuda:0')
        spl_n_state = self.n_state_buf[indx,:].view(-1,self.s_space,1)#.to('cuda:0')
        spl_done = self.done_buf[indx,:].view(-1,1)#.to('cuda:0')

        return spl_c_state, spl_a, spl_rwd, spl_n_state, spl_done

        #return self.buffer[np.random.randint(0,len(self.buffer))]

