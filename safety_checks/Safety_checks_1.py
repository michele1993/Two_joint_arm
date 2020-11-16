import numpy as np
from scipy.integrate import solve_ivp
from Arm_model import Arm_model
from scipy.io import savemat
from safety_checks.Video_arm_config import Video_arm
import torch

# This script creates a matlab file with the result of the python simulation, using "discrete" ivp_solve, with
# a simple constant input, to be compared to the matlab version, after comparison (with a matlab script), obtained
# the same result up to numerical error


npoints = 100
#npoints = 1
tspan = [0, 0.4]
#tspan = [0,0.004]

arm1 = Arm_model(n_points = npoints +1, tspan = tspan) # need the +1 because np.linspace() starts from 0, not from first t_step
# of the interval, so has one fewer value in range than computing 0.4/n_values


# u = np.tile([45,10],npoints).reshape(npoints ,-1)
#
# u = torch.randn(npoints,2) *10
#
# t, thetas = arm1.perfom_reaching(u)
#
#
# end_point = arm1.convert_coord(thetas[-1,0],thetas[-1,1])
#
#
# arm1.plot_info(t, thetas)

# ---------------------------------- TEST IMPLEMENTED RK4 METHOD -------------------------------

u = [300,-100]

t_step = 0.004


t,y = arm1.fixed_RK_4(t_step,u)

t2,y2 = arm1.perfom_reaching(u)



y = np.array(y)


#print(sum(abs(y[-1,:] - y2[-1,:])))


for i in range(npoints+1):
    print(sum(abs(y[i,:] - y2[i,:])))
    #print(sum(abs(y[i,0:4] - y2[i, 0:4])))





# ---------------------------------- UNCOMMENT TO SAVE RESULTS IN MATLAB FILE -----------------

#result = {"py_thetas": thetas, "control_sgl": u.numpy()}
#savemat('discrete_ivp_sol_py_output_Random_input.mat',result) # save result to compare to matlab



# ------------------------------------- UNCOMMENT TO MAKE VIDEO---------------------------

#video1 = Video_arm(arm1, thetas, t)
#video1.make_video()