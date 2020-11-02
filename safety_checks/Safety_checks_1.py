import numpy as np
from scipy.integrate import solve_ivp
from Arm_model import Arm_model
from scipy.io import savemat
from safety_checks.Video_arm_config import Video_arm

# This script creates a matlab file with the result of the python simulation with a simple constant input
# to be compared to the matlab version; after comparision (performed in a matlab script) the matlab and this version
# output the same results up to numerical errors
# Alternatively, this script can be used to create a video of the python simulation with a simple constant input

n_points = 40
arm1 = Arm_model(n_points = n_points)

u = np.tile([45,10],100).reshape(100 ,-1) # need some extra points since solve_ivp iterates more often than the actual evaluation points


t, thetas = arm1.perfom_reaching(u)

end_point = arm1.convert_coord(thetas[-1,0],thetas[-1,1])



arm1.plot_info(t, thetas) # plot result




# ---------------------------------- UNCOMMENT TO SAVE RESULTS IN MATLAB FILE -----------------

#result = {"py_thetas": thetas}
#savemat('New_version_py_sim_output_0.4s.mat',result) # save result to compare to matlab



# ------------------------------------- UNCOMMENT TO MAKE VIDEO---------------------------

#video1 = Video_arm(arm1, thetas, t)
#video1.make_video()