from Arm_model import Arm_model
from Vanilla_Reinf_dynamics.Vanilla_Reinf_Agent import Reinf_Agent
import numpy as np


episodes = 1000
eval_points = 40

arm = Arm_model(n_points = eval_points)
agent = Reinf_Agent(eval_points + 1000) # add some extra params in case simulation runs for longer than eval-points


for ep in range(episodes):

    actions = agent.sample_a() # may need converting to numpy since it's a tensor





    t, thetas = arm.perfom_reaching(actions)




