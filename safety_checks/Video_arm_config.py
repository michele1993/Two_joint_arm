import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Video_arm:

    def __init__(self, arm, thetas,t,fps = 20):


        self.arm = arm
        self.thetas = thetas
        self.t = t

        self.fig, self.ax = plt.subplots()
        self.xy_lim = self.arm.l1 + self.arm.l2

        # needed for FuncAnimation()
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)


    # This method needs to be passed to FuncAnimation to make video i idexed plots
    def animate(self,i):

        self.ax.clear() # clear axis to avoid previous plot still being present
        plt.xlim(-self.xy_lim,self.xy_lim)
        plt.ylim(-self.xy_lim,self.xy_lim)
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("Arm movement")

        t1,t2 = self.thetas[i, 0:2]
        data = self.arm.armconfig_coord(t1,t2)

        trq1,trq2,d_trq1,d_trq2 = self.thetas[i,-4:]

        t = self.t[i]

        labels =['t: '+ str(round(t,3)),'u1: ' + str(round(trq1,3)), 'u2: ' + str(round(trq2,3)), 'd_u1: ' + str(round(d_trq1,3)), 'd_u2: ' + str(round(d_trq2,3))]

        self.ax.plot(data[:,0],data[:,1],color='b')

        self.ax.legend([labels]) # manually re-set labels to avoid having labels of previous plot still present


    def make_video(self):

        frames = len(self.thetas) # set number of frames (i.e. plots)

        ani = matplotlib.animation.FuncAnimation(self.fig,self.animate,frames=frames,repeat=True)
        ani.save("arm_movement.mp4", writer= self.writer)