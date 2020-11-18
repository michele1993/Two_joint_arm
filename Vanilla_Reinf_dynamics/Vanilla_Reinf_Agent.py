import torch
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Normal

# create agent using REINFORCE

class Reinf_Agent(nn.Module): # inherit for easier managing of trainable parameters


    def __init__(self,n, std = 0.5, ln_rate= 0.1, discount = 0.95):

        super().__init__()

        self.discount = discount
        self.std = std
        self.mu_s = nn.Parameter(torch.randn(n-1,2) *10) #*10  #  initalise means randomly, one less, than data points - since itegrate in pairs of values
        self.optimiser = opt.Adam(self.parameters(),ln_rate)
        #self.optimiser = opt.SGD(self.parameters(),ln_rate)
        #print(list(self.parameters())[0] is self.mu_s)





    def sample_a(self): # sample all control signals in one go and store their log p


        d = Normal(self.mu_s, self.std)

        sampled_as = d.sample()

        self.log_ps = d.log_prob(sampled_as)


        return sampled_as


    def update(self, dis_rwd):


        loss = torch.sum(self.log_ps * dis_rwd)# dis_rwd.reshape(-1,1) #.mean() # check that product is element-wise, may need log_ps.view(-1)

        self.optimiser.zero_grad()
        loss.backward()
        # print(list(self.parameters())[0].grad)

        self.optimiser.step()

        self.log_ps = None # flush self.log just in case

        return loss


    # Note: this type of rwd f() may not work for the current problem, the Matlab rwd f() may be a better option
    # due to the rwd being negative and at the final step only and the rest being zero, thus if apply backward discounting
    # initial actions returns will be smaller (thus better) than final actions
    def compute_standard_returns(self,rwd): # compute the correct discounted rwd

        rwds = torch.Tensor(rwd).repeat(len((self.mu_s)))

        discounts = self.discount ** (torch.FloatTensor(range(len(rwds))))

        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts
        # the first bit flip(cumsum(flip(...))) computes the cumulative sum using the discount from the first state,
        # (e.g. rwd for last state super discounted), so then need to divide by the (extra) discounting applied to
        # each state to get right amount of rwd from that state (this work since cumsum is linear, so can
        # take the discounting out of the sum and applying it at the end for the cumsum for each state (very clever!).


    def forward_dis_return(self,rwd ): # forward discounting, like in matlab the closer an a to rwd the more discounted (since rwd is negative)

        n = torch.Tensor(range(len(self.mu_s)))

        return rwd * self.discount**n


    # MAY WANT TO INCLUDE BASELINE!




    # CHECK BACKWARD GRAPH TO SEE WHERE GRADIENT FLOW FROM LOSS ------------------------------
    # print(loss)
    # print(loss.grad_fn.next_functions[0][0])
    # print(loss.grad_fn.next_functions[0][0].next_functions)
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
    # print(
    #     loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
    # print(
    #     loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[
    #         0][0].next_functions)
    # print(
    #     loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[
    #         0][0].next_functions[0][0].next_functions)
    # print(
    #     loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[
    #         0][0].next_functions[0][0].next_functions[0][0].next_functions)
    # print(
    #     loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[
    #         0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0].variable is self.mu_s)