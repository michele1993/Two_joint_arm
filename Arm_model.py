import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp




class Arm_model:

    def __init__(self, height=1.8, mass=80, tspan = [0, 0.4],x0 = [-np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0],n_points = 40):

        # Simulation parameters
        self.tspan = tspan
        self.x0 = x0
        self.eval_points = np.linspace(tspan[0], tspan[1], n_points)


        # Mass and height of ppt
        self.M = mass
        self.L = height

        # mass of upper arm and forearm
        m1 = self.M * 0.028
        m2 = self.M * 0.022

        # length of upper arm and forearm
        self.l1 = self.L * 0.186
        self.l2 = (0.146 + 0.108) * self.L

        # length from center of mass
        lc1 = self.l1 * 0.436
        lc2 = self.l2 * 0.682

        # Inertia
        I1 = m1 * (self.l1 * 0.322) ** 2  # with respect to center of mass of arm
        I2 = m2 * (self.l2 * 0.468) ** 2  # with respect to center of mass of forearm

        self.alpha = m1 * lc1**2 + I1 + m2 * lc2**2 + I2 + m2* self.l1**2
        self.omega = 2 * m2 * self.l1 * lc2

        self.M22 = m2 * lc2**2 + I2


        self.beta = m2 * lc2**2 + I2
        self.delta = m2 * self.l1 * lc2

        np.random.seed(1)
        self.F = np.random.rand(2,2) # viscosity matrix

        #self.F = np.array([[0.4170, 0.0001],[0.7203, 0.3023]]) # VALUES USED BY MATLAB


    def inverse_M(self, theta2): # this methods allows to compute the inverse of matrix (function) M(theta2)

        M11 = self.alpha + self.omega * np.cos(theta2)


        M12 = 0.5 * self.omega * np.cos(theta2) + self.beta


        denom = M11 * self.M22 - M12**2


        return (1. / denom) * np.array([[self.M22, - M12],[-M12, M11]])


    def computeC(self, theta2, d_theta1, d_theta2): # precompute the matrix (function) C(theta2, dtheta1, dtheta2)

        c = self.delta * np.sin(theta2)

        C11 = -2 * d_theta2 * c
        C12 = - d_theta2 * c
        C21 = d_theta1 * c

        return np.array([[C11, C12], [C21, 0]])


    def dynamical_system(self,t,y,u1,u2): # create equivalent 1st order dynamical system of equations to be passed to solve_ivp


        inv_MM = self.inverse_M(y[1])


        CC = self.computeC(y[1],y[2],y[3])

        d_eq = np.dot(inv_MM, ([y[4],y[5]] - np.dot(CC, [y[2],y[3]]) + np.dot(self.F,[y[2],y[3]])))

        dydt = np.array([y[2], y[3], d_eq[0], d_eq[1], y[6], y[7], u1, u2])


        return dydt


    def fixed_RK_4(self, t_step,u):

        n_iterations = int((self.tspan[1] - self.tspan[0]) / t_step)

        # n_iterations = np.copy(t_step)
        # t_step = self.tspan[1] / n_iterations


        t_ = None # pass empty t as not used by the system

        y = []

        c_y = self.x0

        y.append(c_y)

        c_t = 0

        t = []

        t.append(c_t)


        for _ in range(n_iterations):

            # Compute 4 different slopes to perfom the update

            k1 = self.dynamical_system(t_,c_y,u[0],u[1])

            n_y = c_y + (k1 * t_step/2)

            k2 = self.dynamical_system(t_, n_y,u[0],u[1])

            n_y = c_y + (k2 * t_step / 2)

            k3 = self.dynamical_system(t_, n_y, u[0], u[1])

            n_y = c_y + (k3 * t_step)

            k4 = self.dynamical_system(t_, n_y, u[0], u[1])

            c_y += t_step * (1/6) * (k1 + 2*k2 + 2*k3 + k4)

            y.append(c_y.copy()) # otherwise just add memory location and c_y keeps changing so end up with all final values of c_y!!!

            c_t += t_step
            t.append(c_t)

        return t, y




    def perfom_reaching(self, u):

        x0 = self.x0
        t0 = self.tspan[0]
        i = 0
        y = []
        y.append(x0)
        #t_trial = []
        #t_trial.append(t0)


        # run integration one u-input at the time to avoid convergence problem
        for it in self.eval_points[1:]: # select all as upper bound of time interval, but first value which is represented by t0

            #solutions = solve_ivp(self.dynamical_system, [t0, it], x0, args=(u[0], u[1]))
            solutions = solve_ivp(self.dynamical_system, [t0, it], x0, args=(u[i,0], u[i,1]))
            x0 = solutions.y.T[-1,:] # transponse solution for desired format
            y.append(x0) #store last value of integration for that interval, corresponding to the desired time value
            t0 = it
            i+=1
            #t_trial.append(solutions.t.T[-1])





        return self.eval_points, np.array(y) #, t_trial


    def compute_rwd(self,y, x_hat,y_hat): # based on average distance of last five points from target

        [x_c, y_c] = self.convert_coord(y[-1, 0], y[-1, 1])#self.convert_coord(y[-5:, 0], y[-5:, 1])


        return  np.mean(np.sqrt((y_hat - y_c)**2 + (x_hat - x_c)**2))



    # The following methods are useful for computing features of the arm (e.g. position, velocity etc.)

    def convert_coord(self, theta1, theta2):

        t1_t2 = theta1 + theta2
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(t1_t2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(t1_t2)

        return [x, y]



    def armconfig_coord(self, theta1, theta2):


        # if theta is a scalar then can't use len() so set it to single coordinate (x,y)
        if isinstance(theta1, (list, np.ndarray)):

            s_coord = np.zeros((2, len(theta1))) # inialise shoulder position to the origin (0,0) for each configuration

        else:

            s_coord = np.zeros(2)


        e_xcoord = self.l1 * np.cos(theta1)
        e_ycoord = self.l1 * np.sin(theta1)

        t1_t2 = theta1 + theta2

        i_xcoord = self.l1 * np.cos(theta1) + self.l2 * np.cos(t1_t2)
        i_ycoord = self.l1 * np.sin(theta1) + self.l2 * np.sin(t1_t2)

        config = np.array([s_coord, [e_xcoord, e_ycoord], [i_xcoord, i_ycoord]])

        return config


    def xy_velocity(self, t1, t2, dt1, dt2):

        dx = -self.l1 * np.sin(t1) * dt1 - self.l2 * (dt1 + dt2) * np.sin((t1 + t2))
        dy = self.l1 * np.cos(t1) * dt1 + self.l2 * (dt1 + dt2) * np.cos(t1 + t2)

        return dx, dy




    def plot_info(self, t, thetas):


        t = np.array(t)

        theta1 = thetas[:, 0]
        theta2 = thetas[:, 1]
        d_theta1_dt = thetas[:, 2]
        d_theta2_dt = thetas[:, 3]

        fig1 = plt.figure()

        ax1 = fig1.add_subplot(311)
        # ax1.set(xlim=(-0.3, 0.6), ylim=(-0.6,0 ))

        ax2 = fig1.add_subplot(312)
        # ax2.set(xlim=(-0.3, 0.6), ylim=(-0.6, 0))

        ax3 = fig1.add_subplot(313)



        [x, y] = self.convert_coord(theta1, theta2)

        config = self.armconfig_coord(theta1, theta2)


        dx, dy = self.xy_velocity(theta1, theta2, d_theta1_dt, d_theta2_dt)

        vel = np.sqrt(dx**2 + dy**2)
        ax3.plot(t, vel)



        sampled_idx = range(0, len(config[0,0,:]))

        colors = cm.rainbow(np.linspace(0, 1, len(config[0,0,:])))

        for i, c in zip(sampled_idx, colors):
            ax2.plot(config[:, 0, i], config[:, 1, i], color=c)
            ax1.plot(x[i], y[i], 'o', color=c)

        plt.show()






