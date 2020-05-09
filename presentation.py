import os
import numpy as np
import random

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.font_manager as font_manager
from src.util.plottings import laplacian_to_graph
from config import get_config
cf, unparsed = get_config()


def presentation(cf):
    if cf.present=="print_result":
        # results = np.load("./results/results_0320.npy", allow_pickle=True).item()
        results = np.load("./result.npy", allow_pickle=True).item()

        for key in results:
            result = results[key]
            result = np.array(result)
            quants = result[:, 0]
            time = result[:, 1]
            print("Experiment: {} ".format(key) + "mean quant {:.2f} $\pm$ {:.2f}, mean time {:.2f} $\pm$ {:.2f}.".format(
                                                                                                np.mean(quants), np.std(quants),np.mean(time), np.std(time)))
        # import pdb;pdb.set_trace()
    elif cf.present=="plot_bs":
        ############################ batch_size ##############################
        import matplotlib
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-white')
        import numpy as np
        import pandas as pd

        batch_size = np.array([128, 256, 512, 1024, 2048, 4096])
        num_trials = 5
        num_x = batch_size.shape[0]

        results = np.load("./results/result_bs_FINAL.npy", allow_pickle=True).item()

        dic = {}

        for exp_name in results:
            spec = exp_name.split("-")
            if spec[4] != "sgd":
                continue
            width = int(spec[3][4])
            bs = int(spec[5][3:])

            if width not in dic:
                dic[width] = [np.zeros((num_trials,num_x)), np.zeros((num_trials,num_x))]

            ind = np.where(batch_size==bs)[0][0]
            performance = np.array(results[exp_name])[:,0]
            time_elapse = np.array(results[exp_name])[:,1]

            if performance.shape[0] < num_trials:
                print(exp_name, performance.shape[0])
                continue
            if width > 3:
                continue

            dic[width][0][:, ind] = -performance
            dic[width][1][:, ind] = time_elapse

        bound = 1784.89
        fig, ax1 = plt.subplots(figsize=(7,5))
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax1.set_xlabel('Batch Size')
        # ax1.set_ylabel("Approximation Ratio", color='tomato')
        # ax2.set_ylabel("Time Elapsed (sec)", color='cornflowerblue', rotation=270, labelpad=12)
        ax1.set_ylabel("Approximation Ratio")
        ax2.set_ylabel("Time Elapsed (sec)", rotation=270, labelpad=12)

        colors = ['tomato', 'cornflowerblue', 'yellowgreen', 'dimgray']
        ls = ['solid', 'dashdot', 'dotted', 'dashed']

        for i,width in enumerate(dic):
            result_arr = dic[width][0]
            time_arr = dic[width][1]

            result_mean = result_arr.mean(axis=0)/bound
            result_std = result_arr.std(axis=0)/bound/3
            time_mean = time_arr.mean(axis=0)
            time_std = time_arr.std(axis=0)
            
            if np.where(result_mean==0)[0].shape[0] == 0:
                index = num_x
            else:
                index = np.where(result_mean==0)[0][0]
            result_mean = result_mean[:index]
            result_std = result_std[:index]
            time_mean = time_mean[:index]
            time_std = time_std[:index]
            batch_size_s = batch_size[:index]

            ax1.plot(batch_size_s, result_mean, c=colors[i], linestyle=ls[0])
            ax1.fill_between(batch_size_s, result_mean-result_std, result_mean+result_std, color=colors[i], alpha=.12)

            ax2.plot(batch_size_s, time_mean, c=colors[i], linestyle=ls[1])
            ax2.fill_between(batch_size_s, time_mean-time_std, time_mean+time_std, color=colors[i], alpha=.12)

        ax1.plot([], color=colors[0], label="cRBM-1")
        ax1.plot([], color=colors[1], label="cRBM-2")
        ax1.plot([], color=colors[2], label="cRBM-3")


        ax2.plot([], color=colors[-1], linestyle=ls[0], label="Approximation Ratio")
        ax2.plot([], color=colors[-1], linestyle=ls[1], label="Time Elapsed (sec)")

        # ax1.tick_params(axis='y', labelcolor='tomato')
        # ax2.tick_params(axis='y', labelcolor='cornflowerblue')

        ax2.set_xticklabels(batch_size)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xticks(batch_size)
        ax2.set_yticks([100,200,400,800,1600,3200,6400,12800,25600])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines_labels = [ax1.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        legend1 = fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.80, 0.27))

        lines_labels = [ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        legend2 = fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.26, 0.90))
        fig.add_artist(legend2)

        # plt.xscale('log')
        # plt.xticks(batch_size, rotation=45)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.suptitle("Ablation Study on Batch Size", fontsize=14)
        plt.tight_layout(rect=[0, 0.00, 1, 0.95])
        plt.savefig("Ablation_Batch.pdf")
        plt.close()
    elif cf.present=="boxplot":
        ############################ boxplot ##############################
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-white')
        import numpy as np

        nodes = np.array([50, 70, 100, 150, 200, 250])

        results = np.load("./results/main_result2.npy", allow_pickle=True).item()
        data_nnqs = []
        for exp_name in results:
            spec = exp_name.split("-")
            if len(spec)!=8:
                continue
            node = int(spec[2].split(",")[0][1:])
            method = spec[0].split("/")[-1]
            bs = int(spec[5][3:])
            if (method=="netket"):
                ind = np.where(nodes==node)[0][0]
                performance = np.array(results[exp_name])[:,0]
                time_elapse = np.array(results[exp_name])[:,1]
                data_nnqs.append(-performance)

        results = np.load("./results/main_result1.npy", allow_pickle=True).item()
        data_rd = []
        data_gw = []
        data_bm = []
        for exp_name in results:
            spec = exp_name.split("(")
            if len(spec)!=2:
                continue
            node = int(spec[1].split(",")[0])
            method = spec[0]
            if node==300:
                continue
            ind = np.where(nodes==node)[0][0]
            performance = np.array(results[exp_name])[:,0]
            time_elapse = np.array(results[exp_name])[:,1]
            if (method=="random_cut"):
                data_rd.append(performance)
            elif (method=="goemans_williamson"):
                data_gw.append(performance)
            spec = exp_name.split("-")
            method = spec[0].split("/")[-1]
            if (method=="manopt"):
                # bound = np.array(results[exp_name])[:,2]
                data_bm.append(performance)
            

        bound = np.array([216.18, 409.23, 805.41, 1784.89, 3040.00, 4688.24])
        # bound = np.array([216.18, 409.23, 805.41, 1784.89, 3040.00, 4688.24, 6662.38])
        bound = np.expand_dims(bound, -1)
        data_rd = np.stack(data_rd)/bound
        data_gw = np.stack(data_gw)/bound
        # first = data_bm[0][1:]
        # data_bm.remove(data_bm[0])
        # data_bm.append(first)
        data_bm = np.stack(data_bm)/bound
        data_nnqs = np.stack(data_nnqs)/bound
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        plt.figure()

        fig, ax = plt.subplots(figsize=(7,5))

        # # Create a figure instance
        # fig = plt.figure(1)
        # # Create an axes instance
        # ax = fig.add_subplot(111)

        fig.suptitle("Algorithm Performance Comparisons", fontsize=14)

        # Create the boxplot
        # b_rd = ax.boxplot(list(data_rd), positions=np.array(range(6))*4.0-0.9, sym='', widths=0.5)
        b_gw = ax.boxplot(list(data_gw), positions=np.array(range(6))*4.0-0.3, sym='', widths=0.5)
        b_bm = ax.boxplot(list(data_bm), positions=np.array(range(6))*4.0+0.3, sym='', widths=0.5)
        b_nnqs = ax.boxplot(list(data_nnqs), positions=np.array(range(6))*4.0+0.9, sym='', widths=0.5)
        # set_box_color(b_rd, 'yellowgreen') # colors are from http://colorbrewer2.org/
        set_box_color(b_gw, 'cornflowerblue')
        set_box_color(b_bm, 'tomato')
        set_box_color(b_nnqs, 'dimgray')

        ax.title.set_size(10)
        ax.set_xticklabels(nodes)
        ax.tick_params(axis="y", labelsize=10)

        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Approximation Ratio')

        # draw temporary red and blue lines and use them to create a legend
        # plt.plot([], c='yellowgreen', label='RAND')
        # plt.plot([], c='cornflowerblue', label='GW')
        # plt.plot([], c='tomato', label='BM')
        # plt.plot([], c='dimgray', label='qNES')
        # # plt.legend()
        # plt.legend(loc="lower right")

        ax.plot([], c='cornflowerblue', label='GW')
        ax.plot([], c='tomato', label='BM')
        ax.plot([], c='dimgray', label='qNES')
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.90, 0.27))

        plt.tight_layout(rect=[0, 0.00, 1, 0.95])
        plt.savefig('Main_Result.pdf')

        plt.close()




presentation(cf)

############################ lineplot ##############################
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# import numpy as np
# import pandas as pd

# nodes = np.array([50, 70, 90, 100, 150, 200, 250])

# results = np.load("./results.npy", allow_pickle=True).item()
# result_nnqs = np.zeros((10,7))
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
#         time_elapse = np.array(results[exp_name])[:,1]
#         result_nnqs[:, ind] = -performance


# results = np.load("./results/result_0326.npy", allow_pickle=True).item()
# result_rd = np.zeros((10,7))
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
#         time_elapse = np.array(results[exp_name])[:,1]
#         result_rd[:, ind] = -performance
#     elif (method=="goemans_williamson"):
#         ind = np.where(nodes==node)[0][0]
#         performance = np.array(results[exp_name])[:,0]
#         time_elapse = np.array(results[exp_name])[:,1]
#         result_gw[:, ind] = -performance


# bound = np.array([210, 400, 790, 1800, 3000, 4600, 6434])
# result_rd_mean = np.array([149.60, 297.10, 614.30, 1436.80, 2467.30, 3888.00, 5609.30])
# result_gw_mean = np.array([203.40, 380.90, 752.50, 1685.70, 2875.10, 4439.90, 6329.20])
# result_bm_mean = np.array([206.30, 390.90, 776.60, 1719.90, 2931.20, 4526.70, 6434.64])
# result_nnqs_mean = np.array([206.97, 392.98, 777.62, 1721.84, 0, 0, 0])

# result_rd_std = np.array([7.41, 11.48, 16.94, 27.14, 30.27, 39.06, 42.64])
# result_gw_std = np.array([3.61, 8.48, 9.22, 13.10, 22.34, 26.07, 29.75])
# result_bm_std = np.array([0.46, 0.54, 1.56, 1.58, 8.07, 12.96, 9.84])
# result_nnqs_std = np.array([0.01, 0.01, 1.40, 7.21, 0, 0, 0])

# result_rd_mean = -result_rd.mean(axis=0)
# result_gw_mean = -result_gw.mean(axis=0)
# result_nnqs_mean = result_nnqs.mean(axis=0)

# result_rd_std = -result_rd.std(axis=0)
# result_gw_std = -result_gw.std(axis=0)
# result_nnqs_std = result_nnqs.std(axis=0)

# plt.plot(nodes, result_rd_mean, c='m', linestyle='-', label="Random")
# plt.plot(nodes, result_gw_mean, c='g', linestyle='-', label="G&W")
# plt.plot(nodes, result_nnqs_mean, c='b', linestyle='-', label="NQS")
# plt.fill_between(nodes, result_rd_mean-result_rd_std, result_rd_mean+result_rd_std, color='m', alpha=.2)
# plt.fill_between(nodes, result_gw_mean-result_gw_std, result_gw_mean+result_gw_std, color='g', alpha=.2)
# plt.fill_between(nodes, result_nnqs_mean-result_nnqs_std, result_nnqs_mean+result_nnqs_std, color='b', alpha=.2)


# plt.xticks([50, 70, 100, 150, 200, 250], rotation=45)
# # plt.yscale('log')

# plt.suptitle("MaxCut Algorithm Comparison", fontsize=16)
# plt.xlabel('Number of Nodes', fontsize=12)
# plt.ylabel('Cut', fontsize=12)
# plt.legend()
# plt.savefig("linear_plot.pdf")
# plt.close()







# ########################### boxplot ##############################

# # seperate boxes
# fig, axes = plt.subplots(2, int(len(data_rd)/2))
# fig.suptitle("MaxCut - Cut Number for Graphs of Different Sizes", fontsize=14)

# for i,ax in enumerate(axes.reshape(-1)):
#     b_rd = ax.boxplot([data_rd[i]], positions=np.array(range(1))*3.0-0.9, sym='', widths=0.5)
#     b_gw = ax.boxplot([data_gw[i]], positions=np.array(range(1))*3.0-0.3, sym='', widths=0.5)
#     b_bm = ax.boxplot([data_bm[i]], positions=np.array(range(1))*3.0+0.3, sym='', widths=0.5)
#     if i<4:
#         b_nnqs = ax.boxplot([data_nnqs[i]], positions=np.array(range(1))*3.0+0.9, sym='', widths=0.5)
#     set_box_color(b_rd, '#D7191C') # colors are from http://colorbrewer2.org/
#     set_box_color(b_gw, '#2C7BB6')
#     set_box_color(b_bm, '#2C7BB6')
#     set_box_color(b_nnqs, '#2ca25f')
#     ax.set_title('{} Nodes'.format(nodes[i]))
#     ax.title.set_size(10)
#     ax.set_xticks([])
#     ax.tick_params(axis="y", labelsize=6)






# reduce the time by half

# results = np.load("./result.npy", allow_pickle=True).item()
# for exp_name in results:
#     for trails in results[exp_name]:
#         trails[1] = trails[1]/2

# np.save("./result1.npy", results)
# import pdb;pdb.set_trace()