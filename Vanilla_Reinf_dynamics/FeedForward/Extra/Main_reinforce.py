from Arm_model import Arm_model
from Vanilla_Reinf_dynamics.FeedForward.Individual_params.Vanilla_Reinf_Agent import Reinf_Agent
import torch


episodes = 35000
eval_points = 2
t_print = 500

# Target endpoint, based on matlab - reach strainght in fron at shoulder height
x_hat = 0.792
y_hat = 0


arm = Arm_model(n_points = eval_points)
agent = Reinf_Agent(eval_points)

ep_rwds = []
avr_rwd = 0
alpha =  0.01


for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor
    t, thetas = arm.perfom_reaching(actions)

    rwd = torch.Tensor([arm.compute_rwd(thetas,x_hat,y_hat)])


    advantage = rwd - avr_rwd

    avr_rwd += alpha * (rwd - avr_rwd)



    #dis_rwd = torch.Tensor(rwd)
    #dis_rwd = agent.forward_dis_return(rwd)


    #old_mus = torch.clone(agent.mu_s)

    agent.update(advantage)

    #print(sum(old_mus == agent.mu_s))

    ep_rwds.append(rwd)



    if ep % t_print == 0:

        print(sum(ep_rwds)/t_print)
        ep_rwds = []

        #print(agent.mu_s)


test_actions = agent.test_actions(eval_points )

t_t, t_y = arm.perfom_reaching(test_actions)

t_accuracy = torch.Tensor([arm.compute_rwd(t_y,x_hat,y_hat)])

print("Test Accuracy: ", t_accuracy)