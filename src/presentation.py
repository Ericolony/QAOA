import os
import numpy as np
import random
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.font_manager as font_manager
from src.util.plottings import laplacian_to_graph
from config import get_config
cf, unparsed = get_config()



def presentation(cf):
    if cf.present=="print_result":
        results = np.load("./result.npy", allow_pickle=True).item()
        for key in results:
            result = results[key]
            result = np.array(result)
            scores = result[:, 0]
            time = result[:, 1]
            print("Experiment: {} ".format(key) + "mean score {:.2f} $\pm$ {:.2f}, mean time {:.2f} $\pm$ {:.2f}.".format(
                    np.abs(np.mean(scores)), np.abs(np.std(scores)),np.mean(time), np.std(time)))
    elif cf.present=="plot_bs":
        ############################ batch_size ##############################
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
        plt.savefig("Ablation_Batch.png")
        plt.close()
    elif cf.present=="plot_box":
        ############################ boxplot ##############################
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
        plt.savefig('Main_Result.png')

        plt.close()

presentation(cf)