import torch



class V_Memory_B:

    def __init__(self, a_space,n_arms,dev,s_space = 2,batch_size = 30,n_targets=50,size = 10000):

        self.size = size
        self.batch_size = batch_size
        self.s_space = s_space
        self.a_space = a_space
        self.n_arms = n_arms
        self.n_targets = n_targets

        self.c_size = 0
        self.dev = dev

        # Intialise Tensor buffer for each compotent on CPU to avoid running out of memory
        self.target_state_buf = torch.zeros((self.size,s_space)).to(self.dev)
        self.actions_buf = torch.zeros((self.size,a_space)).to(self.dev)
        self.rwd_buf = torch.zeros((self.size,1)).to(self.dev)


    def store_transition(self,c_state,action,rwd):

        # Determine indx for buffer, based on ratio n of stored steps vs size
        c_idx = self.c_size % self.size # this goes back to 0 whenever buffer completed
        f_idx = c_idx + (self.n_arms * self.n_targets)

        self.target_state_buf[c_idx:f_idx,:] = c_state.repeat(self.n_arms,1)
        self.actions_buf[c_idx:f_idx,:] = action
        self.rwd_buf[c_idx:f_idx,:] = rwd

        self.c_size += (self.n_arms * self.n_targets)


    def sample_transition(self):

        indx_upper_b = min(self.c_size,self.size) # check if buffer is full and take appropriate indx
        indx = torch.randint(0,indx_upper_b,(self.batch_size,)).to(self.dev)

        # Sample corresponding transition
        spl_t_state = self.target_state_buf[indx,:]
        spl_a = self.actions_buf[indx,:]
        spl_rwd = self.rwd_buf[indx,:]


        return spl_t_state, spl_a, spl_rwd