from TD_3.FeedBack.TD3_FB_Actor_Critic import *
import torch
import torch.nn as nn

class TD3:

    def __init__(self,actor,critic1,critic2,buffer,decay_upd,n_arms,dev, t_policy_noise=0.2,t_noise_clip=0.5 ,actor_update=2,discount=1):

        self.dev = dev
        self.discount = torch.tensor(discount).to(self.dev)
        self.decay_upd = torch.tensor(decay_upd).to(self.dev)
        self.actor_update = actor_update
        self.t_pol_noise = t_policy_noise
        self.t_clip_noise = t_noise_clip

        self.actor = actor
        #self.actor.apply(self.xavier_w_init)
        self.actor.apply(self.small_weight_init)

        self.critic_1 = critic1
        self.critic_1.apply(self.xavier_w_init)
        # self.critic_1.apply(self.small_weight_init)

        self.critic_2 = critic2
        self.critic_2.apply(self.xavier_w_init)
        # self.critic_2.apply(self.small_weight_init)

        # Initialise Memory Buffer
        self.MBuffer = buffer

        self.batch_s = buffer.batch_size

        lamb = critic1.lamb

        # Initialise the first target critic target NN
        self.critic_target_1 = Critic_NN(lamb,dev).to(self.dev)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict()) # Make sure two critic NN have the same initial parameters
        self.critic_target_1.freeze_params()# Freeze the critic target NN parameter

        # Initialise the second target critic target NN
        self.critic_target_2 = Critic_NN(lamb, dev).to(self.dev)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict()) # Make sure two critic NN have the same initial parameters
        self.critic_target_2.freeze_params()# Freeze the critic target NN parameter

        #Do the same for actor
        self.target_agent = Actor_NN(n_arms).to(self.dev)
        self.target_agent.load_state_dict(self.actor.state_dict())
        self.target_agent.freeze_params()


    def xavier_w_init(self, l):

        if type(l) == nn.Linear:
            nn.init.xavier_normal_(l.weight, gain=0.000005)
            l.bias.data.fill_(0)

    # small initialisation

    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std=0.0001)
            nn.init.normal_(l.bias,mean=0,std=.001)


    def update(self,step):

        # Randomly sample batch of transitions from buffer
        spl_c_state, spl_a, spl_rwd, spl_n_state, spl_done = self.MBuffer.sample_transition()

        # Create input for target critic, based on next state and the optimal action there
        optimal_a = self.target_agent(spl_n_state) # compute optimal greedy action for the next state

        tot_batch_size = optimal_a.size()

        # Add noise to optimal action:
        a_noise = (torch.randn(tot_batch_size) * self.t_pol_noise).to(self.dev)
        a_noise = torch.cat([a_noise[:,0:2].clamp(-self.t_clip_noise,self.t_clip_noise), a_noise[:,2:].clamp(0,self.t_clip_noise)],dim=1)
        optimal_a = optimal_a + a_noise # based on stable baselines hyper-params

        # Select min target for each batch
        target = torch.min(self.critic_target_1(spl_n_state,optimal_a), self.critic_target_2(spl_n_state,optimal_a))

        # Compute two Q target value
        Q_target = spl_rwd + spl_done * self.discount * target  # estimate maxQ given optimal action at next state in reply

        # if torch.sum(torch.isnan(spl_a)) >0:
        #     print('spl_a')
        #     print(spl_a)
        #     exit()
        #
        # if torch.sum(torch.isnan(spl_c_state)) >0:
        #     print('spl_state')
        #     print(spl_c_state)
        #     exit()

        # Compute Q estimate based on reply episode
        Q_estimate_1 = self.critic_1(spl_c_state,spl_a)
        Q_estimate_2 = self.critic_2(spl_c_state, spl_a)

        # if torch.sum(torch.isnan(Q_estimate_1)) >0:
        #     print('q1')
        #     print(spl_a[torch.isnan(Q_estimate_1).squeeze(),:],'\n')
        #     print(spl_c_state[torch.isnan(Q_estimate_1).squeeze(),:], '\n')
        #     print(spl_c_state)
        #
        #     # for p in self.critic_1.parameters():
        #     #     print(torch.sum(torch.isnan(p.data)))
        #     exit()


        # Update critic
        critic_loss1 = self.critic_1.update(Q_target, Q_estimate_1)
        critic_loss2 = self.critic_2.update(Q_target, Q_estimate_2)

        # if torch.sum(torch.isnan(critic_loss1)) >0:
        #     print('loss1')
        #     print(Q_estimate_1,'\n')


        actor_loss = torch.tensor(0)


        # Update actor based on first critic
        if step % self.actor_update == 0:

            # select smaller batch for actor
            Q_size = Q_estimate_1.size()[0]
            intervalns = torch.linspace(0, Q_size, self.batch_s + 1)[:-1]
            rand_values = torch.randint(self.batch_s - 1, (self.batch_s,))
            actor_indx = (intervalns + rand_values).long()

            actor_s = spl_c_state[actor_indx]

            #actor_loss = self.critic_1(spl_c_state, self.actor(spl_c_state)) # loss for actor based on first critic only
            actor_loss = self.critic_1(actor_s, self.actor(actor_s))

            actor_loss = self.actor.update(actor_loss)

            # Update target NN through polyak average
            self.target_agent.soft_update(self.actor, self.decay_upd)

        self.critic_target_1.soft_update(self.critic_1, self.decay_upd)
        self.critic_target_2.soft_update(self.critic_2, self.decay_upd)


        return critic_loss1,critic_loss2, actor_loss