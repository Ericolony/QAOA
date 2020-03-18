import os
import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
from src.train import run_netket, run_pyket
from src.offshelf.maxcut import off_the_shelf
from src.util.helper import record_result

def main(cf):
    # set up directories
    prepare_dirs_and_logger(cf)

    # set up data
    data = load_data(cf)

    # run with netket/flowket
    if cf.framework == 'netket':
        exp_name, quant, time_ellapsed = run_netket(cf, data)
    elif cf.framework == 'flowket':
        if cf.pyket_on_cpu:
            with tf.device('/cpu:0'):
                time_in_seconds = run_pyket(cf, data)
        else:
            time_in_seconds = run_pyket(cf, data)
    elif cf.framework == "random_cut":
        exp_name, quant, time_ellapsed = off_the_shelf(cf, laplacian=data, method="random_cut")
    elif cf.framework == "greedy_cut":
        exp_name, quant, time_ellapsed = off_the_shelf(cf, laplacian=data, method="greedy_cut")
    elif cf.framework == "goemans_williamson":
        exp_name, quant, time_ellapsed = off_the_shelf(cf, laplacian=data, method="goemans_williamson")
    else:
        raise Exception('unknown framework')
    return exp_name, quant, time_ellapsed


if __name__ == '__main__':
    cf, unparsed = get_config()
    for num_trials in range(cf.num_trials):
        seed = cf.random_seed + num_trials
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        random.seed(seed)

        exp_name, quant, time_ellapsed = main(cf)
        record_result(cf, exp_name, quant, time_ellapsed)
    print('finished')