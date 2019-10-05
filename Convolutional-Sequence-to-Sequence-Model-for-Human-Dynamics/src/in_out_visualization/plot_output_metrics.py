import numpy as np

a = np.load('output_for_linearizedloss_CNN.npy')
print(a)
# from matplotlib import pyplot as plt
# a_avg = np.mean(a,axis=0)
#
# print('Mean')
# print(a_avg)
# axiss = np.arange(a_avg.shape[0])
# plt.plot(axiss, a_avg )
# plt.show()