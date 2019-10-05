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
import os

def plot_animation_seq(input_seq, predict, xyz_pred_gt, filename, make_images):
	predict_plot = plot_h36m_seq(input_seq, predict, xyz_pred_gt, filename, make_images)
	return predict_plot

class plot_h36m_seq(object):

	def __init__(self, input_seq, predict, pred_gt, filename, make_images):
		self.make_images = make_images
		self.joint_xyz = input_seq
		self.nframes_in = input_seq.shape[0]
		self.joint_xyz_out = predict
		self.joint_xyz_out_gt = pred_gt
		self.nframes_out = predict.shape[0]
		self.nframes = self.nframes_in + self.nframes_out

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
		if(frame<self.nframes_in):
			xdata = np.squeeze(self.joint_xyz[frame, :, 0])
			ydata = np.squeeze(self.joint_xyz[frame, :, 1])
			zdata = np.squeeze(self.joint_xyz[frame, :, 2])

			for i in range(len(self.chain)):
				self.lns.append(
					self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],],
					               linewidth=2.0, color='#0780ea'))  # blue: input

		else:
			frame_to_consider = frame - self.nframes_in
			xdata = np.squeeze(self.joint_xyz_out[frame_to_consider, :, 0])
			ydata = np.squeeze(self.joint_xyz_out[frame_to_consider, :, 1])
			zdata = np.squeeze(self.joint_xyz_out[frame_to_consider, :, 2])
			xdata_gt = np.squeeze(self.joint_xyz_out_gt[frame_to_consider, :, 0])
			ydata_gt = np.squeeze(self.joint_xyz_out_gt[frame_to_consider, :, 1])
			zdata_gt = np.squeeze(self.joint_xyz_out_gt[frame_to_consider, :, 2])

			for i in range(len(self.chain)):
				self.lns.append(
					self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],],
					               linewidth=2.0, color='#DA0000'))  # Red: prediction
				self.lns.append(
						self.ax.plot3D(xdata_gt[self.chain[i][:],], ydata_gt[self.chain[i][:],], zdata_gt[self.chain[i][:],],
		               linewidth=2.0, color='#0000DA'))  # Blue: prediction truth

		# this saves image
		if(self.make_images):
			path = './images_output/' + self.filename
			if(os.path.exists(path) == False):
				os.mkdir(path)
			plt.savefig(path+'/'+str(frame) + ".png")



	def plot(self):
		ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
		plt.title(self.filename, fontsize=16)
		ani.save(self.filename + '.gif', writer='imagemagick')
		# plt.show()

