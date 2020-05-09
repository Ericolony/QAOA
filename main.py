
import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.directory import prepare_dirs_and_logger
from src.util.data_loader import load_data
from src.util.helper import record_result

from src.train import run_netket
from src.offshelf.MaxCut import off_the_shelf
from src.offshelf.manopt_maxcut import manopt
from RL.train import train


def main(cf, seed):
    # set up directories
    prepare_dirs_and_logger(cf)

    # set up data
    data = load_data(cf)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))
    if cf.framework in ["netket"]:
        exp_name, score, time_elapsed = run_netket(cf, data, seed)
    elif cf.framework in ["random_cut", "greedy_cut", "goemans_williamson"]:
        exp_name, score, time_elapsed = off_the_shelf(cf, laplacian=data, method=cf.framework)
    elif cf.framework in ["manopt"]:
        exp_name, score, time_elapsed, bound = manopt(cf, laplacian=data)
    elif cf.framework in ["RL"]:
        exp_name, score, time_elapsed = train(cf, data)
    else:
        raise Exception("unknown framework")
    return exp_name, score, time_elapsed, bound


if __name__ == '__main__':
    cf, unparsed = get_config()
    for num_trials in range(cf.num_trials):
        seed = cf.random_seed + num_trials
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        random.seed(seed)

        exp_name, score, time_elapsed, bound = main(cf, seed)
        record_result(cf, exp_name, score, time_elapsed, bound)
    print('finished')