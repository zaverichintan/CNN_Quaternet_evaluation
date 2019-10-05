import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import scipy.io as sio
import qforward_kinematics as FK
import numpy.random as rnd
from plot_animation import plot_animation

import sys

def get_data():

	# in_daata = sio.loadmat(path+'/input_files/gt_lie_'+action+'_'+num+'.mat')['gt']
	y_pred = sio.loadmat('prediction_quat.mat')['prediction']
	y_gt = sio.loadmat('gt_quat.mat')['gt']


	return y_gt, y_pred

# main

gt, pred = get_data()
print(gt.shape)

bone_length = np.array(
	[0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
	 -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
	 0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
	 0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000, 257.077681,
	 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000,
	 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000, 0.000000, 0.000000, 0.000000,
	 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 257.077681,
	 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924, 0.000000, 0.000000, 251.728680, 0.000000, 0.000000,
	 0.000000, 0.000000, 0.000000, 0.000000, 99.999888, 0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                   17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

fk = FK.create_batch_tf_calculate(parent, bone_length)

gt_tf = tf.convert_to_tensor(gt, dtype='float32')
pred_tf = tf.convert_to_tensor(pred, dtype='float32')
gt_xyz = fk(gt_tf)
pred_xyz = fk(pred_tf)
gt_numpy = gt_xyz.numpy()
pred_numpy = pred_xyz.numpy()

filename = 'Quat_to_xyz'
predict_plot = plot_animation(pred_numpy, gt_numpy, filename)
predict_plot.plot()
