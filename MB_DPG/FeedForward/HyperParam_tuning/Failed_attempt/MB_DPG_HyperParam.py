import torch
from MB_DPG.FeedForward.HyperParam_tuning.Failed_attempt.AccuracyStorage import _AccStorage



# ISSUE: code reproduce correctly only results for first set of params, but then already for the second
# set of params, results change, suggesting something of previous iteration, affected the result

#seed = 1
#torch.manual_seed(seed)

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


dev = torch.device('cpu')


_range_ln_rate = torch.linspace(0.00001,0.01,10).to(dev) #[0.0078] #5.5600e-03  #  0.0078# #[1.0000e-05] #[5.5600e-03]#torch.linspace(0.00001,0.01,10)
_range_ln_rate_2 = torch.linspace(0.00001,0.01,10).to(dev) #[0.0011] # #[1.1200e-03] #[1.0000e-05] #[4.4500e-03]

_storage = _AccStorage(n1=len(_range_ln_rate), n2=len(_range_ln_rate_2))
#range_std = torch.linspace(0.001,0.05,10)#0.01


_i = 0 # initialise counter across all loops
for _model_ln  in _range_ln_rate: #ln_rate_a
    #for std in range_std:
    for _actor_ln in _range_ln_rate_2: #range_ln_rate

        for name in dir():
            if not name.startswith('_'):
                del globals()[name]

        from MB_DPG.FeedForward.HyperParam_tuning.MB_DPG_train import MB_DPG_train
        import torch
        import numpy as np
        import random

        #torch.use_deterministic_algorithms(True)
        #torch.set_deterministic(True)
        std = 0.0124
        episodes = 5001
        seed = 1
        dev = torch.device('cpu')

        torch.manual_seed(seed)  # re-set seeds everytime to ensure same initialisation
        np.random.seed(seed)
        random.seed(seed)


        # redefine everything at each iteration to avoid potential memory leakages
        MBDPG = MB_DPG_train(_model_ln, _actor_ln, std, episodes, dev)
        training_acc, training_vel = MBDPG.train()
        _storage.add(_i,training_acc,training_vel, _model_ln, _actor_ln)

        _i += 1

        print("iteration n: ", _i)
        print("_model_ln", _model_ln)
        print("_actor_ln", _actor_ln)
        print("Accuracy: ", training_acc)
        print("Velocity: ", training_vel)

print("One value")
_storage.save_data()