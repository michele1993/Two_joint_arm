import torch
#import torch.distributions.multinomial as multinomial
class MemBuffer:

# Attempts to implement a prioritised replay buffer to train the model only
    def __init__(self,n_arms,a_size,est_y_size,dev, size = 5000):

        self.dev = dev
        self.n_arms = n_arms
        self.size = size
        self.n_stored_samples = 0

        self.rwd_buffer = torch.zeros(self.size).to(self.dev) # store exp(rwd) action and outcome
        self.action_buffer = torch.zeros((self.size,a_size)).to(self.dev)
        self.endState_buffer = torch.zeros((self.size,est_y_size)).to(self.dev)

        self.probs = torch.zeros(self.size).to(self.dev)

        self.gamma = 1e-8

        self.epsilon = 0#0.75 # 0.8 # 0.7

    def store(self, rwds, actions, outcomes):

        exp_rwds = torch.exp(-rwds)  # need to perform some computation so that smaller rwds are valued more

        # works as long as n_arms/size = integer
        c_idx = self.n_stored_samples % self.size

        up_idx = c_idx + self.n_arms

        self.rwd_buffer[c_idx:up_idx] = exp_rwds

        self.action_buffer[c_idx:up_idx,:] = actions
        self.endState_buffer[c_idx:up_idx,:] = outcomes

        self.probs = self.rwd_buffer.clone() / torch.sum(self.rwd_buffer) + self.gamma # add a very small constant to avoid p < 0

        self.n_stored_samples += self.n_arms


    def sample(self,n_samples):

        # select number of stored sampled so far or size of buffer if n. of store sampled is greater
        indx_upper_b = min(self.n_stored_samples, self.size)


        # To avoid model being too much biased towards high reward experience, which may induce high velocity
        # use e-greedy method, where sampled from prioritised buffer with p = epsilon and sampled
        # from a uniform buffer with p = 1 - epsilon

        rand_v = torch.rand(1).to(self.dev)

        if rand_v < self.epsilon:

            # sample upto the point at which buffer has been filled
            indx = self.probs[:indx_upper_b].multinomial(num_samples=n_samples, replacement=True).to(self.dev)

        else:

            indx = torch.randint(0, indx_upper_b, (n_samples,)).to(self.dev)

        return self.action_buffer[indx,:], self.endState_buffer[indx,:]