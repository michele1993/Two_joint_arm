import numpy as np
import torch




class Q_Par_Arm_model:

    def __init__(self,t_step,x0,goal,t_window,dev, n_arms=10, height=1.8, mass=80):

        self.dev = dev
        # Simulation parameters
        self.t_step = t_step
        self.n_arms = n_arms
        self.t_window = t_window


        # Target parameters
        self.x_hat, self.y_hat = goal

        # State parameters
        self.x0 = torch.Tensor(x0).repeat(self.n_arms, 1, 1).to(self.dev)
        self.c_x = x0

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



    def dynamical_system(self,y,u,c_decay): # create equivalent 1st order dynamical system of equations to be passed to solve_ivp

        inv_MM = self.inverse_M(y[:,1])

        CC = self.computeC(y[:,1],y[:,2],y[:,3])

        d_thet = y[:,2:4]

        torques = y[:,4:6]

        eq_rhs = torques - CC @ d_thet + self.F @ d_thet

        d_eq = inv_MM @ eq_rhs


        return torch.cat([d_thet, d_eq, y[:,6:8] - c_decay * torques,u - c_decay * y[:,6:8]],dim=1)



    def step(self, action,t):

        action = torch.unsqueeze(action,dim=2) # need to increase dim to fit dynamical system parallel operations
        u = action[:,0:2]
        c_decay = action[:,2:3]

        # Compute 4 different slopes to perfom the update

        k1 = self.dynamical_system(self.c_x,u,c_decay) # use it:it+1 to keep the dim of original tensor without slicing it

        n_y = self.c_x + (k1 * self.t_step/2)

        k2 = self.dynamical_system( n_y,u,c_decay)

        n_y = self.c_x + (k2 * self.t_step / 2)

        k3 = self.dynamical_system( n_y, u,c_decay)

        n_y = self.c_x + (k3 * self.t_step)

        k4 = self.dynamical_system( n_y, u,c_decay)

        self.c_x = self.c_x + self.t_step * (1/6) * (k1 + 2*k2 + 2*k3 + k4) # use w_average of the for slope to compute a slope from initial point

        if t <= self.t_window:

            return self.c_x, torch.zeros(1).expand(self.n_arms,1).to(self.dev), torch.zeros(1).expand(self.n_arms,1).to(self.dev)
        else:

            return self.c_x, self.compute_distance(self.c_x), self.compute_vel(self.c_x)

    def reset(self):

        self.c_x = self.x0

        return self.x0

    def compute_distance(self,y): # based on average distance of last five points from target

        [x_c, y_c] = self.convert_coord(y[:, 0], y[:, 1])

        return (self.y_hat - y_c)**2 + (self.x_hat - x_c)**2


    def compute_vel(self,y):

        t1 = y[:,0]
        t2 = y[:,1]
        dt1 = y[:,2]
        dt2 = y[:,3]

        dx = - self.l1 * torch.sin(t1) * dt1 - self.l2 * (dt1+dt2) * torch.sin((t1+t2 ))
        dy = self.l1 * torch.cos(t1) * dt1 + self.l2 * (dt1 + dt2) * torch.cos((t1 + t2))

        return dx**2 + dy**2

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