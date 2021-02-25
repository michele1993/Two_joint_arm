import random
import torch


class V_Memory_B:

    def __init__(self,batch_size = 1,size = 20000):

        self.size = size
        self.batch_size = batch_size

        self.buffer = []
        self.c_size = 0


    def store_transition(self,c_state,action,rwd,n_s,dn):


        # Check if the replay buffer is full
        if len(self.buffer) <= self.size:

            self.buffer.append((c_state,action,rwd,n_s,dn))

        # if full, start replacing values from the first element
        else:

            self.buffer[self.c_size] = (c_state,action,rwd,n_s,dn)
            self.c_size+=1

            # Need to restart indx when reach end of list
            if self.c_size == self.size:
                self.c_size = 0



    def sample_transition(self):

        # Randomly sample some batches of transition,each containing n_arms transitions
        spl_transitions = random.sample(self.buffer, self.batch_size)
        spl_c_state, spl_a, spl_rwd, spl_n_state, spl_done = zip(*spl_transitions)

        # Concatenate each tuple in one big tensor
        spl_c_state = torch.cat(spl_c_state)
        spl_a = torch.cat(spl_a)
        spl_rwd = torch.cat(spl_rwd)
        spl_n_state = torch.cat(spl_n_state)
        spl_done = torch.cat(spl_done)

        return spl_c_state, spl_a, spl_rwd, spl_n_state, spl_done

        #return self.buffer[np.random.randint(0,len(self.buffer))]