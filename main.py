import os
import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
from src.util.helper import record_result

def main(cf):
    # set up directories
    prepare_dirs_and_logger(cf)

    # set up data
    data = load_data(cf)

    bound = None
    # run with netket/flowket
    if cf.framework == 'netket':
        from src.train import run_netket, run_pyket
        exp_name, quant, time_ellapsed = run_netket(cf, data)
    elif cf.framework == 'flowket':
        from src.train import run_pyket
        if cf.pyket_on_cpu:
            with tf.device('/cpu:0'):
                exp_name, quant, time_ellapsed = run_pyket(cf, data)
        else:
            exp_name, quant, time_ellapsed = run_pyket(cf, data)
    elif cf.framework in ["random_cut", "greedy_cut", "goemans_williamson", "sdp_BM", "sdp_SCS", "sdp_CVXOPT", "debug"]:
        from src.offshelf.MaxCut import off_the_shelf
        exp_name, quant, time_ellapsed = off_the_shelf(cf, laplacian=data, method=cf.framework)
    elif cf.framework in ["manopt"]:
        from src.offshelf.manopt_maxcut import manopt
        exp_name, quant, time_ellapsed, bound = manopt(cf, laplacian=data)
    elif cf.framework == "RL":
        from RL.train import train
        exp_name, quant, time_ellapsed = train(cf, data)
    else:
        raise Exception('unknown framework')
    return exp_name, quant, time_ellapsed, bound


if __name__ == '__main__':
    cf, unparsed = get_config()
    for num_trials in range(cf.num_trials):
        cf.random_seed = cf.random_seed + num_trials
        np.random.seed(cf.random_seed)
        tf.random.set_random_seed(cf.random_seed)
        random.seed(cf.random_seed)

        exp_name, quant, time_ellapsed, bound = main(cf)
        record_result(cf, exp_name, quant, time_ellapsed, bound)
    print('finished')