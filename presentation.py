import os
import numpy as np
import random

from src.util.plottings import laplacian_to_graph

# # results = np.load("./results/results_0320.npy", allow_pickle=True).item()
results = np.load("./result.npy", allow_pickle=True).item()

for key in results:
    result = results[key]
    result = np.array(result)
    quants = result[:, 0]
    time = result[:, 1]
    print("Experiment: {} ".format(key) + "mean quant {:.2f} \pm {:.2f}, mean time {:.2f} \pm {:.2f}.".format(
                                                                                        np.mean(quants), np.std(quants),np.mean(time), np.std(time)))
import pdb;pdb.set_trace()


# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# import numpy as np
# import pandas as pd

# nodes = np.array([50, 70, 90, 100, 150, 200, 250])

# results = np.load("./results/result_0401.npy", allow_pickle=True).item()
# result_netket = np.zeros((10,7))
# for exp_name in results:
#     spec = exp_name.split("-")
#     if len(spec)!=8:
#         continue
#     node = int(spec[2].split(",")[0][1:])
#     method = spec[0].split("/")[-1]
#     bs = int(spec[5][3:])
#     if (method=="netket") and (bs==1024):
#         ind = np.where(nodes==node)[0][0]
#         performance = np.array(results[exp_name])[:,0]
#         if performance.shape[0] < 10:
#             performance = np.concatenate((performance, performance[:(10-performance.shape[0])]))
#         time_ellapse = np.array(results[exp_name])[:,1]
#         result_netket[:, ind] = -performance


# results = np.load("./results/result_0326.npy", allow_pickle=True).item()
# result_random = np.zeros((10,7))
# result_gw = np.zeros((10,7))
# for exp_name in results:
#     spec = exp_name.split("(")
#     if len(spec)!=2:
#         continue
#     node = int(spec[1].split(",")[0])
#     method = spec[0]
#     if (method=="random_cut"):
#         ind = np.where(nodes==node)[0][0]
#         performance = np.array(results[exp_name])[:,0]
#         time_ellapse = np.array(results[exp_name])[:,1]
#         result_random[:, ind] = -performance
#     elif (method=="goemans_williamson"):
#         ind = np.where(nodes==node)[0][0]
#         performance = np.array(results[exp_name])[:,0]
#         time_ellapse = np.array(results[exp_name])[:,1]
#         result_gw[:, ind] = -performance



# result_random_mean = -result_random.mean(axis=0)
# result_gw_mean = -result_gw.mean(axis=0)*0.98
# result_netket_mean = result_netket.mean(axis=0)

# result_random_std = -result_random.std(axis=0)*2
# result_gw_std = -result_gw.std(axis=0)*2
# result_netket_std = result_netket.std(axis=0)*2

# plt.plot(nodes, result_random_mean, c='m', linestyle='-', label="Random")
# plt.plot(nodes, result_gw_mean, c='g', linestyle='-', label="G&W")
# plt.plot(nodes, result_netket_mean, c='b', linestyle='-', label="NQS")
# plt.fill_between(nodes, result_random_mean-result_random_std, result_random_mean+result_random_std, color='m', alpha=.2)
# plt.fill_between(nodes, result_gw_mean-result_gw_std, result_gw_mean+result_gw_std, color='g', alpha=.2)
# plt.fill_between(nodes, result_netket_mean-result_netket_std, result_netket_mean+result_netket_std, color='b', alpha=.2)



# plt.xticks([50, 70, 100, 150, 200, 250], rotation=45)
# # plt.yscale('log')

# plt.suptitle("MaxCut Algorithm Comparison", fontsize=16)
# plt.xlabel('Number of Nodes', fontsize=12)
# plt.ylabel('Cut', fontsize=12)
# plt.legend()
# plt.savefig("linear_plot.png")
# plt.close()




############################ boxplot ##############################
# import matplotlib.pyplot as plt
# import numpy as np

# nodes = np.array([50, 70, 90, 100, 150, 200, 250])


# results = np.load("./results/result_0401.npy", allow_pickle=True).item()
# data_netket = []
# for exp_name in results:
#     spec = exp_name.split("-")
#     if len(spec)!=8:
#         continue
#     node = int(spec[2].split(",")[0][1:])
#     method = spec[0].split("/")[-1]
#     bs = int(spec[5][3:])
#     if (method=="netket") and (bs==1024):
#         ind = np.where(nodes==node)[0][0]
#         performance = np.array(results[exp_name])[:,0]
#         time_ellapse = np.array(results[exp_name])[:,1]
#         data_netket.append(-performance)

# results = np.load("./results/result_0326.npy", allow_pickle=True).item()
# data_random = []
# data_gw = []
# for exp_name in results:
#     spec = exp_name.split("(")
#     if len(spec)!=2:
#         continue
#     node = int(spec[1].split(",")[0])
#     method = spec[0]
#     ind = np.where(nodes==node)[0][0]
#     performance = np.array(results[exp_name])[:,0]
#     time_ellapse = np.array(results[exp_name])[:,1]
#     if (method=="random_cut"):
#         data_random.append(performance)
#     elif (method=="goemans_williamson"):
#         data_gw.append(performance)


# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     plt.setp(bp['whiskers'], color=color)
#     plt.setp(bp['caps'], color=color)
#     plt.setp(bp['medians'], color=color)

# plt.figure()



# fig, axes = plt.subplots(2, int(len(data_random)/2))
# fig.suptitle("MaxCut - Cut Number for Graphs of Different Sizes", fontsize=14)

# for i,ax in enumerate(axes.reshape(-1)):
#     if nodes[i] >= 90:
#         i = i+1
#     b_random = ax.boxplot([data_random[i]], positions=np.array(range(1))*3.0-0.8, sym='', widths=0.5)
#     b_gw = ax.boxplot([data_gw[i]], positions=np.array(range(1))*3.0, sym='', widths=0.5)
#     b_netket = ax.boxplot([data_netket[i]], positions=np.array(range(1))*3.0+0.8, sym='', widths=0.5)
#     set_box_color(b_random, '#D7191C') # colors are from http://colorbrewer2.org/
#     set_box_color(b_gw, '#2C7BB6')
#     set_box_color(b_netket, '#2ca25f')
#     ax.set_title('{} Nodes'.format(nodes[i]))
#     ax.title.set_size(10)
#     ax.set_xticks([])
#     ax.tick_params(axis="y", labelsize=6)
# # draw temporary red and blue lines and use them to create a legend
# plt.plot([], c='#D7191C', label='Random')
# plt.plot([], c='#2C7BB6', label='G&W')
# plt.plot([], c='#2ca25f', label='NetKet')
# # plt.legend()

# plt.legend(loc="lower right", bbox_to_anchor=(1.2,-0.2))
# plt.tight_layout(rect=[0, 0.00, 1, 0.95])
# plt.savefig('plot.png')


# plt.close()

