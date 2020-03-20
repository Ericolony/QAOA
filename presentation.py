import os
import numpy as np
import random

results = np.load("./results/results_0320.npy", allow_pickle=True).item()

for key in results:
    result = results[key]
    result = np.array(result)
    quants = result[:, 0]
    time = result[:, 1]
    print("Experiment: {} ".format(key) + "mean quant {:.2f} \pm {:.2f}, mean time {:.2f} \pm {:.2f}.".format(
                                                                                        np.mean(quants), np.std(quants),np.mean(time), np.std(time)))
import pdb;pdb.set_trace()