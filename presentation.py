import os
import numpy as np
import random
import tensorflow as tf

from config import get_config
from main import main

cf, unparsed = get_config()
np.random.seed(cf.random_seed)
tf.random.set_random_seed(cf.random_seed)

time_in_seconds = main(cf)
print('finished')
print('%s iterations take %s seconds' % (cf.num_of_iterations, time_in_seconds))
f=open(os.path.join(cf.dir, "result.txt"), "a+")
f.write("{} iterations take {:.2f} seconds\n".format(cf.num_of_iterations, time_in_seconds))
f.write("----------------------------------------------------------------------------------------\n")
f.close()