#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_animation(predict, labels, filename):
    predict_plot = plot_h36m(predict, labels, filename)
    return predict_plot

class plot_h36m(object):

    def __init__(self, predict, labels, filename):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -750
        xmax = 750
        ymin = -750
        ymax = 750
        zmin = -750
        zmax = 750

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.chain = [np.array([0, 1, 2, 3, 4, 5]),
                             np.array([0, 6, 7, 8, 9, 10]),
                             np.array([0, 12, 13, 14, 15]),
                             np.array([13, 17, 18, 19, 22, 19, 21]),
                             np.array([13, 25, 26, 27, 30, 27, 29])]
        self.scats = []
        self.lns = []
        self.filename = filename

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        plt.title(self.filename, fontsize=16)
        ani.save(self.filename + '.gif', writer='imagemagick')
        # plt.show()

