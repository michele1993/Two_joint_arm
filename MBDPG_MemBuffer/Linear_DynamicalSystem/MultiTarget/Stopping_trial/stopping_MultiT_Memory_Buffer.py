import torch

class MemBuffer:


    def __init__(self,n_arms,a_size,est_y_size,window_s,dev, size = 5000):

        self.dev = dev
        self.n_arms = n_arms
        self.size = size
        self.n_stored_samples = 0

        self.action_buffer = torch.zeros((self.size,a_size)).to(self.dev)
        self.endState_buffer = torch.zeros((window_s,self.size,est_y_size)).to(self.dev)


    def store(self,  actions, outcomes): #rwds


        c_idx = self.n_stored_samples % self.size

        up_idx = c_idx + self.n_arms

        self.action_buffer[c_idx:up_idx,:] = actions
        self.endState_buffer[:,c_idx:up_idx,:] = outcomes

        self.n_stored_samples += self.n_arms


    def sample(self,n_samples):

        # select number of stored sampled so far or size of buffer if n. of store sampled is greater
        indx_upper_b = min(self.n_stored_samples, self.size)
        indx = torch.randint(0, indx_upper_b, (n_samples,)).to(self.dev)

        return self.action_buffer[indx,:], self.endState_buffer[:,indx,:]