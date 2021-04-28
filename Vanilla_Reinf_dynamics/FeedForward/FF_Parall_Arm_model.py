import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import torch
from math import pi, cos, sin
from random import random




class Parall_Arm_model:

    def __init__(self,tspan,x0,dev, n_arms=10, height=1.8, mass=80):

        self.dev = dev
        # Simulation parameters
        self.tspan = tspan
        self.n_arms = n_arms

        #self.x0 = torch.Tensor(x0).expand(self.n_arms,-1,-1) # -1 leaves dim unchaged, for 1d tensor, always use expand instead of repeat

        #I fear that by sharing memory location of x0(i.e. with expand()) may get some weird unpredicted error, better not risk it
        self.x0 = torch.Tensor(x0).repeat(self.n_arms, 1, 1).to(self.dev)

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

        M22 = torch.Tensor([m2 * lc2 ** 2 + I2]).to(self.dev)
        self.M22 = M22.repeat(self.n_arms,1)

        self.C22 = torch.Tensor([[0]]).expand(self.n_arms, 1).to(self.dev)

        self.beta = m2 * lc2**2 + I2
        self.delta = m2 * self.l1 * lc2

        np.random.seed(1)
        self.F = torch.Tensor(np.random.rand(2,2)).repeat(n_arms,1,1).to(self.dev) *15 # repeat in first dimension given n of arms



    def inverse_M(self, theta2): # this methods allows to compute the inverse of matrix (function) M(theta2)



        M11 = self.alpha + self.omega * torch.cos(theta2)

        M12 = 0.5 * self.omega * torch.cos(theta2) + self.beta

        denom = M11 * self.M22 - M12 ** 2

        return ((1 / denom) * torch.cat([self.M22, -M12, -M12, M11],dim=1)).reshape(-1,2,2) # make 2x2xn Tensor with each corresponding entry


    def computeC(self, theta2, d_theta1, d_theta2): # precompute the matrix (function) C(theta2, dtheta1, dtheta2)

        c = self.delta * torch.sin(theta2)

        C11 = -2 * torch.mul(d_theta2, c)
        C12 = torch.mul(- d_theta2, c)
        C21 = torch.mul(d_theta1, c)


        return torch.cat([C11, C12, C21, self.C22],dim=1).reshape(-1,2,2)



    def dynamical_system(self,y,u): # create equivalent 1st order dynamical system of equations to be passed to solve_ivp

        inv_MM = self.inverse_M(y[:,1])

        CC = self.computeC(y[:,1],y[:,2],y[:,3])

        d_thet = y[:,2:4]

        torques = y[:,4:6]

        eq_rhs = torques - CC @ d_thet + self.F @ d_thet

        d_eq = inv_MM @ eq_rhs

        return torch.cat([d_thet, d_eq, y[:,6:8],u],dim=1)


    # if torch.sum(torch.isnan(y)) >0: # check if y contains any nan
    #     print('y')
    #     print(y)
    #     exit()



    def perform_reaching(self, t_step,u):

        n_iterations = int((self.tspan[1] - self.tspan[0]) / t_step)

        y = []
        c_y = self.x0.clone() # need to detach ? No if you wanna differentiate through RK
        c_t = 0
        t = []
        t.append(c_t)


        for it in range(n_iterations):

            y.append(c_y.detach().clone()) # store intermediate values, but without keeping track of gradient for each

            # Compute 4 different slopes, k, for initial point, to perform one-step update according to RK4 implementation

            k1 = self.dynamical_system(c_y,u[:,:,it:it+1]) # use it:it+1 to keep the dim of original tensor without slicing it

            n_y = c_y + (k1 * t_step/2)

            k2 = self.dynamical_system( n_y,u[:,:,it:it+1])

            n_y = c_y + (k2 * t_step / 2)

            k3 = self.dynamical_system( n_y, u[:,:,it:it+1])

            n_y = c_y + (k3 * t_step)

            k4 = self.dynamical_system( n_y, u[:,:,it:it+1])

            #c_y += t_step * (1/6) * (k1 + 2*k2 + 2*k3 + k4) # use w_average of the for slope to compute a slope from initial point
            c_y = c_y + t_step * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            c_t += t_step

            t.append(c_t)

        y.append(c_y) # store final locations, which contains backward gradient, through all previous points

        return t, torch.stack(y)


    def compute_rwd(self,y, x_hat,y_hat, f_points): # based on average distance of last five points from target

        #[x_c, y_c] = self.convert_coord(y[-1:, :,0], y[-1:,:, 1])
        [x_c, y_c] = self.convert_coord(y[f_points:, :, 0], y[f_points:, :, 1])

        return (x_hat - x_c)**2 + (y_hat - y_c)**2 #torch.sqrt()# maintain original dimension for product with log_p


    # Generate n random points along the circumference reacheable by the arm
    def circof_random_tagrget(self,n):

        radius = self.l1 + self.l2
        points = []
        quadrants = [-pi/2, pi/2] # define for

        for i in range(n):
            count = i % 2
            theta = random() * quadrants[count]
            x = radius * cos(theta)
            y = radius * sin(theta)
            points.append([x,y])

        # Plot to show if points correctly generated
        # plt.scatter(*zip(*points))
        # plt.show()

        return torch.tensor(points).to(self.dev)

    # Use this method to compute rwd when have multiple tatget x_hat, y_hat
    def multiP_compute_rwd(self,y, x_hat,y_hat, f_points,target_n_arms): # n_arms for each target, different from self.n_arms which refers to over all n of arms

        #[x_c, y_c] = self.convert_coord(y[-1:, :,0], y[-1:,:, 1])
        [x_c, y_c] = self.convert_coord(y[f_points:, :, 0], y[f_points:, :, 1])

        # repeat target for each arm, matching x_c and y_c dimensions
        x_hat = x_hat.repeat(1,target_n_arms,1)
        y_hat = y_hat.repeat(1,target_n_arms,1)

        return (x_hat - x_c)**2 + (y_hat - y_c)**2


    def compute_vel(self,y, f_points):

        t1 = y[f_points:, :,0] # [-1:,
        t2 = y[f_points:, :,1] # [-1:,
        dt1 = y[f_points:, :,2] # [-1:,
        dt2 = y[f_points:, :,3] # [-1:,

        dx = - self.l1 * torch.sin(t1) * dt1 - self.l2 * (dt1+dt2) * torch.sin((t1+t2 ))
        dy = self.l1 * torch.cos(t1) * dt1 + self.l2 * (dt1 + dt2) * torch.cos((t1 + t2))


        return dx**2 + dy**2 #torch.sqrt() # maintain original dimension to sum with rwd

    def compute_accel(self, vel, t_step):

        return (vel[1:, :, :] - vel[:-1, :, :]) / t_step


    # The following methods are useful for computing features of the arm (e.g. position, velocity etc.)

    def convert_coord(self, theta1, theta2):

        t1_t2 = theta1 + theta2
        x = self.l1 * torch.cos(theta1) + self.l2 * torch.cos(t1_t2)
        y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(t1_t2)

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
