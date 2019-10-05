from __future__ import division

import numpy as np
import h5py
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D
# import viz
# import time
import copy
import data_utils
import tensorflow as tf
import os
from tempfile import TemporaryFile
from plot_animation import plot_animation


#define which action to visualize
tf.app.flags.DEFINE_string("action",'all', "specify action to visualize")
# tf.app.flags.DEFINE_string("file", "CNNAdTrain_GANWEIGHT0.000000_Sampleing0.950000_samples.h5", "which file to load")
tf.app.flags.DEFINE_string("file", "CNNAdTrain_GANWEIGHT0.010000_Sampleing0.950000WindownLength20-24000.h5", "which file to load")



# tf.app.flags.DEFINE_string("sub_index", "_7", "which seed to visualize")
tf.app.flags.DEFINE_string("vis", "gt", "which seed to visualize")
FLAGS = tf.app.flags.FLAGS


def fkl( angles, parent, offset, rotInd, expmapInd ):
	"""
	Convert joint angles and bone lenghts into the 3d points of a person.
	Based on expmap2xyz.m, available at
	https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

	Args
	  angles: 99-long vector with 3d position and 3d joint angles in expmap format
	  parent: 32-long vector with parent-child relationships in the kinematic tree
	  offset: 96-long vector with bone lenghts
	  rotInd: 32-long list with indices into angles
	  expmapInd: 32-long list with indices into expmap angles
	Returns
	  xyz: 32x3 3d points that represent a person in 3d space
	"""

	assert len(angles) == 99

	# Structure that indicates parents for each joint
	njoints   = 32
	xyzStruct = [dict() for x in range(njoints)]

	for i in np.arange( njoints ):

		if not rotInd[i] : # If the list is empty
			xangle, yangle, zangle = 0, 0, 0
		else:
			xangle = angles[ rotInd[i][0]-1 ]
			yangle = angles[ rotInd[i][1]-1 ]
			zangle = angles[ rotInd[i][2]-1 ]


		r = angles[ expmapInd[i] ]

		thisRotation = data_utils.expmap2rotmat(r)
		thisPosition = np.array([xangle, yangle, zangle])

		if parent[i] == -1: # Root node
			xyzStruct[i]['rotation'] = thisRotation
			xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
		else:
			xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
			xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

	xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
	xyz = np.array( xyz ).squeeze()
	xyz = xyz[:,[0,2,1]]
	# xyz = xyz[:,[2,0,1]]


	return xyz

def revert_coordinate_space(channels, R0, T0):
	"""
	Bring a series of poses to a canonical form so they are facing the camera when they start.
	Adapted from
	https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

	Args
	  channels: n-by-99 matrix of poses
	  R0: 3x3 rotation for the first frame
	  T0: 1x3 position for the first frame
	Returns
	  channels_rec: The passed poses, but the first has T0 and R0, and the
					rest of the sequence is modified accordingly.
	"""
	n, d = channels.shape

	channels_rec = copy.copy(channels)
	R_prev = R0
	T_prev = T0
	rootRotInd = np.arange(3,6)

	# Loop through the passed posses
	for ii in range(n):
		R_diff = data_utils.expmap2rotmat( channels[ii, rootRotInd] )
		R = R_diff.dot( R_prev )

		channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
		T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
		channels_rec[ii,:3] = T
		T_prev = T
		R_prev = R

	return channels_rec


def _some_variables():
	"""
	We define some variables that are useful to run the kinematic tree

	Args
	  None
	Returns
	  parent: 32-long vector with parent-child relationships in the kinematic tree
	  offset: 96-long vector with bone lenghts
	  rotInd: 32-long list with indices into angles
	  expmapInd: 32-long list with indices into expmap angles
	"""

	parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
	                   17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

	offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
	offset = offset.reshape(-1,3)

	rotInd = [[5, 6, 4],
	          [8, 9, 7],
	          [11, 12, 10],
	          [14, 15, 13],
	          [17, 18, 16],
	          [],
	          [20, 21, 19],
	          [23, 24, 22],
	          [26, 27, 25],
	          [29, 30, 28],
	          [],
	          [32, 33, 31],
	          [35, 36, 34],
	          [38, 39, 37],
	          [41, 42, 40],
	          [],
	          [44, 45, 43],
	          [47, 48, 46],
	          [50, 51, 49],
	          [53, 54, 52],
	          [56, 57, 55],
	          [],
	          [59, 60, 58],
	          [],
	          [62, 63, 61],
	          [65, 66, 64],
	          [68, 69, 67],
	          [71, 72, 70],
	          [74, 75, 73],
	          [],
	          [77, 78, 76],
	          []]

	expmapInd = np.split(np.arange(4,100)-1,32)

	return parent, offset, rotInd, expmapInd

def main():
	if FLAGS.action=='all':
		actions = ["walking", "eating", "smoking", "discussion",  "directions",
		           "greeting", "phoning", "posing", "purchases", "sitting",
		           "sittingdown", "takingphoto", "waiting", "walkingdog",
		           "walkingtogether"]
	else:
		actions=FLAGS.action

	number_of_subsequence = 256
	over_all_error_metric = []
	# Load all the data
	parent, offset, rotInd, expmapInd = _some_variables()

	# number_of_frames = 32
	number_of_frames = 100
	l2_metric = np.zeros(number_of_frames, )

	for action in actions:
		print(action)
		for seed in np.arange(number_of_subsequence):
			# print(seed)

			with h5py.File( '/media/data/zaveri/CNN/samples_100/'+FLAGS.file, 'r' ) as h5f:

				expmap_gt = h5f['expmap/gt/' + action + '_'+str(seed)][50:, :]

				if FLAGS.vis=='preds':
					expmap_pred = h5f['expmap/preds/' + action + '_'+str(seed)][:]
				else:
					expmap_pred = h5f['expmap/preds/' + action + '_' + str(seed)][:, :]

			nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

			# Put them together and revert the coordinate space
			expmap_all = revert_coordinate_space( np.vstack((expmap_gt, expmap_pred)), np.eye(3), np.zeros(3) )
			expmap_gt = expmap_all[:nframes_gt,:]
			expmap_pred = expmap_all[nframes_gt:,:]

			# # trick
			expmap_gt[:, :6] = 0
			expmap_pred[:, :6] = 0

			# Compute 3d points for each frame
			xyz_gt, xyz_pred = np.zeros((nframes_gt, 32, 3)), np.zeros((nframes_pred, 32, 3))
			for i in range( nframes_gt ):
				xyz_gt[i, :] = fkl( expmap_gt[i, :], parent, offset, rotInd, expmapInd)
			for i in range( nframes_pred ):
				xyz_pred[i, :] = fkl( expmap_pred[i, :], parent, offset, rotInd, expmapInd)
			print(xyz_gt.shape)
			l2_metric += l2(xyz_gt, xyz_pred)

			#save 3D poses
			np.save('./3dposes/pred'+ '_'+ action+'_'+str(seed)+'.npy', xyz_pred)
			np.save('./3dposes/gt' + '_' + action + '_' + str(seed) + '.npy', xyz_gt)
		#
		# l2_metric = l2_metric / number_of_subsequence
		# l2_metric = l2_metric / 100
		# l2_metric_rounded = np.around(l2_metric, decimals=4)
		# toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
		# idx = np.where(toprint_idx < len(l2_metric))[0]
		# print(idx)
		# toprint_list = ["& {:.2f} ".format(l2_metric[toprint_idx[i]]) for i in idx]
		# print("".join(toprint_list))

		# print(l2_metric_rounded)
	# 	over_all_error_metric.append(l2_metric_rounded)
	# print(over_all_error_metric)
	#
	# avg = np.mean(over_all_error_metric, axis=0)
	# toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
	# idx = np.where(avg < len(avg))[0]
	# toprint_list = ["& {:.2f} ".format(avg[toprint_idx[i]]) for i in idx]
	# print("".join(toprint_list))



# print("#################################")
	# over_all_error_metric_np = np.array(over_all_error_metric)
	# print(over_all_error_metric_np)
	# outfile = TemporaryFile()
	# np.save('output_for_linearizedloss_CNN', over_all_error_metric_np)


def l2(gt=None, pred=None):
	A = gt - pred
	A = A ** 2
	l2_metric = np.sqrt(np.sum(A, axis=2))

	l2_metric = np.sum(l2_metric, axis=1) / A.shape[1]
	return l2_metric


if __name__ == '__main__':
	main()
