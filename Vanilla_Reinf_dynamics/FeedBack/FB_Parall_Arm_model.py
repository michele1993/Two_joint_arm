import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch




class FB_Par_Arm_model:

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



    def perform_reaching(self, t_step,agent,train):

        n_iterations = int((self.tspan[1] - self.tspan[0]) / t_step)

        y = []
        c_y = self.x0.clone() # need to detach ? No if you wanna differentiate through RK
        u_values = []


        for it in range(n_iterations):

            u = agent(c_y,train)
            u_values.append(u)

            y.append(c_y.detach().clone()) # store intermediate values, but without keeping track of gradient for each

            # Compute 4 different slopes to perfom the update

            k1 = self.dynamical_system(c_y,u) # use it:it+1 to keep the dim of original tensor without slicing it

            n_y = c_y + (k1 * t_step/2)

            k2 = self.dynamical_system( n_y,u)

            n_y = c_y + (k2 * t_step / 2)

            k3 = self.dynamical_system( n_y, u)

            n_y = c_y + (k3 * t_step)

            k4 = self.dynamical_system( n_y, u)

            c_y += t_step * (1/6) * (k1 + 2*k2 + 2*k3 + k4) # use w_average of the for slope to compute a slope from initial point


        y.append(c_y) # store final locations, which contains backward gradient, through all previous points

        return torch.stack(y), torch.stack(u_values)




    def compute_rwd(self,y, t1_hat,t2_hat): # based on average distance of last five points from target

        # Based on the openAI inverted pendulum

        t1 = (t1_hat - y[1:,:,0])**2 # don't take first value since based on initial cond

        # NEED TO SOME TWO ANGLES FOR SECOND TORQUE ?
        t2 = (t2_hat - y[1:, :, 1]) ** 2 # don't take first value since based on initial cond

        dt1 = y[1:, :, 2] ** 2 # don't take first value since based on initial cond
        dt2 = y[1:, :, 3] ** 2 # don't take first value since based on initial cond

        return t1 + t2 + 0.1*(dt1+dt2)

    def compute_distance(self,y, x_hat,y_hat): # based on average distance of last five points from target

        [x_c, y_c] = self.convert_coord(y[-1, :, 0], y[-1, :, 1])

        return torch.sqrt((y_hat - y_c)**2 + (x_hat - x_c)**2)



    def compute_vel(self,y, f_points):

        t1 = y[-1, :,0] # [-1:,
        t2 = y[-1, :,1] # [-1:,
        dt1 = y[-1, :,2] # [-1:,
        dt2 = y[-1, :,3] # [-1:,

        dx = - self.l1 * torch.sin(t1) * dt1 - self.l2 * (dt1+dt2) * torch.sin((t1+t2 ))
        dy = self.l1 * torch.cos(t1) * dt1 + self.l2 * (dt1 + dt2) * torch.cos((t1 + t2))

        return torch.sqrt(dx**2 + dy**2)



    # The following methods are useful for computing features of the arm (e.g. position, velocity etc.)

    def convert_coord(self, theta1, theta2):

        t1_t2 = theta1 + theta2
        x = self.l1 * torch.cos(theta1) + self.l2 * torch.cos(t1_t2)
        y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(t1_t2)

        return [x, y]