import os
import numpy as np
import random
import tensorflow as tf

from config import get_config
from src.util.directory import prepare_dirs_and_logger
from src.util.helper import load_data
from src.train import run_netket, run_pyket


def main(cf):
    # set up directories
    prepare_dirs_and_logger(cf)

    # set up data
    data = load_data(cf)

    # run with netket/flowket
    if cf.framework == 'netket':
        time_in_seconds = run_netket(cf, data)
    elif cf.framework == 'flowket':
        if cf.pyket_on_cpu:
            with tf.device('/cpu:0'):
                time_in_seconds = run_pyket(cf, data)
        else:
            time_in_seconds = run_pyket(cf, data)
    else:
        raise Exception('unknown framework')
    return time_in_seconds


if __name__ == '__main__':
    cf, unparsed = get_config()
    np.random.seed(cf.random_seed)
    tf.random.set_random_seed(cf.random_seed)

    time_in_seconds = main(cf)
    print('finished')
    print('%s iterations take %s seconds' % (cf.num_of_iterations, time_in_seconds))
    f=open(os.path.join(cf.dir, "result.txt"), "a+")
    f.write("{} iterations take {} seconds\n".format(cf.num_of_iterations, time_in_seconds))
    f.write("----------------------------------------------------------------------------------------\n")
    f.close()