import numpy as np
from scipy.integrate import solve_ivp
from Arm_model import Arm_model
from scipy.io import savemat
from safety_checks.Video_arm_config import Video_arm

# This script creates a matlab file with the result of the python simulation with a simple constant input
# to be compared to the matlab version

npoints = 40
arm1 = Arm_model(n_points = npoints)

u = [45, 5]

arm1 = Arm_model()



t, thetas = arm1.perfom_reaching(u)


end_point = arm1.convert_coord(thetas[-1,0],thetas[-1,1])

print(t.shape)

arm1.plot_info(t, thetas)




# ---------------------------------- UNCOMMENT TO SAVE RESULTS IN MATLAB FILE -----------------

#result = {"py_thetas": thetas}
#savemat('python_sim_output_1s.mat',result) # save result to compare to matlab



# ------------------------------------- UNCOMMENT TO MAKE VIDEO---------------------------

#video1 = Video_arm(arm1, thetas, t)
#video1.make_video()