from DDPG.DDPG_FB_Actor_Critic import *

class DDPG:

    def __init__(self,actor,critic,buffer,decay_upd,dev,discount=0.99):

        self.dev = dev
        self.discount = torch.tensor(discount).to(self.dev)
        self.decay_upd = torch.tensor(decay_upd).to(self.dev)

        self.actor = actor
        self.critic = critic

        # Initialise Memory Buffer
        self.MBuffer = buffer

        # Initialise a target critic target NN
        self.critic_target = Critic_NN().to(self.dev)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Make sure two critic NN have the same initial parameters
        self.critic_target.freeze_params()# Freeze the critic target NN parameter

        #Do the same for actor
        self.target_agent = Actor_NN().to(self.dev)
        self.target_agent.load_state_dict(self.actor.state_dict())
        self.target_agent.freeze_params()




    def update(self):

        # Randomly sample batch of transitions from buffer
        spl_c_state, spl_a, spl_rwd, spl_n_state, spl_done = self.MBuffer.sample_transition()

        # Create input for target critic, based on next state and the optimal action there
        optimal_a = self.target_agent(spl_n_state) # compute optimal greedy action for the next state

        # Compute Q target value
        Q_target = spl_rwd + spl_done * self.discount * self.critic_target(spl_n_state,optimal_a) # estimate maxQ given optimal action at next state in reply

        # Compute Q estimate based on reply episode
        Q_estimate = self.critic(spl_c_state,spl_a)

        # Update critic
        critic_loss = self.critic.update(Q_target, Q_estimate)

        # Update actor
        actor_loss = self.critic(spl_c_state, self.actor(spl_c_state))

        self.actor.update(actor_loss)

        # Update target NN through polyak average
        self.target_agent.soft_update(self.actor, self.decay_upd)
        self.critic_target.soft_update(self.critic, self.decay_upd)

        return critic_loss