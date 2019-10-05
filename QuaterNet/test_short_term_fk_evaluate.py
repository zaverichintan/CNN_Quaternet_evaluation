# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from common.mocap_dataset import MocapDataset
from common.quaternion import qeuler_np
from short_term.pose_network_short_term import PoseNetworkShortTerm
from short_term.dataset_h36m import dataset, subjects_test, short_term_weights_path
import numpy as np
import scipy.io as sio
import tensorflow as tf
tf.enable_eager_execution()
import scipy.io as sio
import qforward_kinematics as FK

torch.manual_seed(1234)

def find_indices_srnn(data, action, subject, num_seeds, prefix_length, target_length):
    """
    This method replicates the behavior of the same method in
    https://github.com/una-dinosauria/human-motion-prediction
    """
    # Same seed as in https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    rnd = np.random.RandomState(1234567890)

    # A subject performs the same action twice in the H3.6M dataset
    T1 = data[(subject, action, 1)].shape[0]
    T2 = data[(subject, action, 2)].shape[0]

    idx = []
    for i in range(num_seeds//2):
        idx.append(rnd.randint(16, T1-prefix_length-target_length))
        idx.append(rnd.randint(16, T2-prefix_length-target_length))
    return idx

def build_sequence_map_srnn(data):
    """
    This method replicates the behavior of the same method in
    https://github.com/una-dinosauria/human-motion-prediction
    """
    out = {}
    for subject in data.subjects():
        for action, seq in data[subject].items():
            if not '_d0' in action or '_m' in action:
                continue
            act, sub, _ = action.split('_')
            out[(int(subject[1:]), act, int(sub))] = seq['rotations']
    return out

def get_test_data(data, action, subject, num_seeds):
    """
    This method replicates the behavior of the same method in
    https://github.com/una-dinosauria/human-motion-prediction
    """
    seq_map = build_sequence_map_srnn(data)
    # num_seeds = 8 # Eight test sequences for each action, as in SRNN
    # num_seeds = 256 # 256 test sequences for each action, as in SRNN

    prefix_length = 50
    target_length = 100 # We don't actually use all 100 frames
    indices = find_indices_srnn(seq_map, action, subject, num_seeds, prefix_length, target_length)
    seeds = [(action, (i%2)+1, indices[i]) for i in range(num_seeds)]
    out = []
    for i in range(num_seeds):
        _, subsequence, idx = seeds[i]
        idx = idx + 50
        chunk = seq_map[(subject, action, subsequence)]
        chunk = chunk[(idx-prefix_length):(idx+target_length) ,:]
        out.append((chunk[0:prefix_length-1, :],
                    chunk[prefix_length-1:(prefix_length+target_length-1), :],
                    chunk[prefix_length:, :]))
    return out
    
def evaluate(model, test_data, num_seeds):
    bone_length = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681,
         0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681, 0.000000,
         0.000000,
         151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000, 0.000000, 0.000000,
         0.000000,
         0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         257.077681,
         0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924, 0.000000, 0.000000, 251.728680, 0.000000,
         0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 99.999888, 0.000000, 137.499922, 0.000000, 0.000000, 0.000000,
         0.000000])
    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    number_of_frames = 32
    number_of_subsequence = num_seeds
    l2_metric = np.zeros(number_of_frames, )

    for d in test_data:
        source = np.concatenate((d[0], d[1][:1]), axis=0).reshape(-1, 32*4)
        target = d[2].reshape(-1, 32*4)
        if model is None:
            target_predicted = np.tile(source[-1], target.shape[0]).reshape(-1, 32*4)
        else:
            target_predicted = model.predict(np.expand_dims(source, 0), target_length=np.max(frame_targets)+1).reshape(-1, 32*4)
            
        # target = qeuler_np(target[:target_predicted.shape[0]].reshape(-1, 4), 'zyx').reshape(-1, 96)
        # target_predicted = qeuler_np(target_predicted.reshape(-1, 4), 'zyx').reshape(-1, 96)

        fk = FK.create_batch_tf_calculate(parent, bone_length)
        gt = target[:target_predicted.shape[0]]
        pred = target_predicted
        gt = np.reshape(gt, [gt.shape[0], -1, 4])
        pred = np.reshape(pred, [pred.shape[0], -1, 4])

        sio.savemat('prediction_quat.mat', dict([('prediction', pred)]))
        sio.savemat('gt_quat.mat', dict([('gt', gt)]))

        gt_tf = tf.convert_to_tensor(gt, dtype='float32')

        pred_tf = tf.convert_to_tensor(pred, dtype='float32')
        gt_xyz = fk(gt_tf)
        pred_xyz = fk(pred_tf)
        gt_numpy = gt_xyz.numpy()
        pred_numpy = pred_xyz.numpy()

        # to reduce the frames
        y_p_xyz = gt_numpy[:number_of_frames]
        y_t_xyz = pred_numpy[:number_of_frames]

        l2_metric += l2(y_t_xyz, y_p_xyz)

    l2_metric = l2_metric / number_of_subsequence
    l2_metric = l2_metric / 1000

    # e = np.sqrt(np.sum((target_predicted[:, 3:] - target[:, 3:])**2, axis=1))
    #     errors.append(e)
    # errors = np.mean(np.array(errors), axis=0)
    return l2_metric


def l2(gt=None, pred=None):
    A = gt - pred
    A = A ** 2
    A = A + 0.0000001
    l2_metric = np.sqrt(np.sum(A, axis=2))

    l2_metric = np.sum(l2_metric, axis=1) / A.shape[1]
    return l2_metric


def print_results(action, errors):
    # print(action)
    # for f, e in zip(frame_targets, errors[frame_targets]):
    #     print((f+1)/25*1000, 'ms:', e)
    # print()
    mme = errors
    print("\n" + action)
    toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
    idx = np.where(toprint_idx < len(mme))[0]
    toprint_list = ["& {:.3f} ".format(mme[toprint_idx[i]]) for i in idx]
    print("".join(toprint_list))

frame_targets = [1, 3, 7, 9, 14, 19, 24, 49, 74, 99] # 80, 160, 320, and 400 ms (at 25 Hz)
all_errors = np.zeros((15, 100))

def run_evaluation(model=None):
        for subject_test in subjects_test:
            print('Testing on subject', subject_test)
            print()
            num_seeds = 256
            for idx, action in enumerate(['walking', 'eating', 'smoking', 'discussion',
                      'directions', 'greeting', 'phoning', 'posing', 'purchases',
                      'sitting', 'sittingdown', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']):
                test_data = get_test_data(dataset, action, int(subject_test[1:]), num_seeds)
                errors = evaluate(model, test_data, num_seeds)
                # all_errors[idx] = errors
                print_results(action, errors)


if __name__ == '__main__':
    model = PoseNetworkShortTerm(prefix_length=50)
    if torch.cuda.is_available():
        model.cuda()
    model.load_weights(short_term_weights_path)
    model.eval()
    run_evaluation(model)