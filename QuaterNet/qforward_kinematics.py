import numpy as np
import tensorflow as tf


def qrot(q, v):
	"""
	:param q: [n_batch, 4]
	:param v: [n_batch, 3]
	"""
	qvec = q[:, 1:]

	uv = tf.linalg.cross(qvec, v)
	uuv = tf.linalg.cross(qvec, uv)
	return v + 2 * (q[:, :1] * uv + uuv)


def qmul(q, r):
	"""
	:param q: [n_batch, 4]
	:param r: [n_batch, 4]
	"""
	q = tf.expand_dims(q, 2)
	r = tf.expand_dims(r, 1)
	terms = tf.matmul(q, r)

	w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
	x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
	y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
	z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

	return tf.stack((w, x, y, z), axis=1)


def create_batch_tf_calculate(parent, bone_length):
	"""
	:param parent: [n_joints] list of parent node indices
	:bone_length: [n_joints x 3] defines the bone length
	"""

	chain_per_joint = []
	for jid in range(len(parent)):
		current = parent[jid]
		chain = [current]
		while current > -1:
			current = parent[current]
			chain.append(current)
		chain.reverse()
		chain.pop(0)
		chain_per_joint.append(chain)

	bone_length = np.reshape(bone_length, (-1, 3)).astype('float32')
	n_joints = len(parent)
	dim = len(parent) * 3

	flip_axis = np.array([0, 2, 1], dtype=np.int64)


	def calculate(quaternions):
		"""
		:param [n_batch x n_joints x 4]
		"""
		n_batch = tf.shape(quaternions)[0]
		Pts3d = []

		for jid in range(n_joints):
			q = quaternions[:, jid]
			bone = bone_length[jid]
			bone = tf.tile(bone, multiples=[n_batch])
			bone = tf.reshape(bone, (n_batch, 3))
			chain = chain_per_joint[jid]

			p_xyz = tf.zeros((n_batch, 3), dtype=tf.float32)
			p_q = tf.constant([1, 0, 0, 0], dtype=tf.float32)
			p_q = tf.tile(p_q, multiples=[n_batch])
			p_q = tf.reshape(p_q, (n_batch, 4))

			for jid2 in chain:
				cur_q = quaternions[:, jid2]
				cur_bone = bone_length[jid2]
				cur_bone = tf.tile(cur_bone, multiples=[n_batch])
				cur_bone = tf.reshape(cur_bone, (n_batch, 3))

				p_xyz = p_xyz + qrot(p_q, cur_bone)
				p_q = qmul(cur_q, p_q)

			xyz = qrot(p_q, bone) + p_xyz
			xyz = tf.reshape(xyz, (n_batch, 3))
			Pts3d.append(xyz)

		Pts3d = tf.stack(Pts3d, axis=1)
		return Pts3d

	return calculate