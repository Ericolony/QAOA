import time
import os
import scipy.io
import numpy as np


def manopt(cf, laplacian):
    inputs = {'inputs':laplacian}
    if not os.path.exists("./manopt/saved"):
        os.makedirs("./manopt/saved")
    input_file_name = "./manopt/saved/graph{}.mat".format(cf.input_size)
    scipy.io.savemat(input_file_name, inputs)
    output_file_name = "./manopt/saved/output.mat"

    start_time = time.time()
    matlab_cmd = "cd manopt;matlab -nodisplay -nosplash -nodesktop -r 'run {} .{} .{}, exit(0)';cd ..".format(cf.random_seed,input_file_name,output_file_name)
    os.system(matlab_cmd)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("---matlab takes %s seconds ---" % (time_elapsed))

    outputs = scipy.io.loadmat(output_file_name)
    sol, quant, bound, time_elapsed = outputs["x"], outputs["cutvalue"][0][0], outputs["cutvalue_upperbound"][0,0], outputs["totaltime"][0,0]
    # exp_name, sep, tail = (cf.dir).partition('-date')
    exp_name = cf.framework + str(cf.input_size)
    return exp_name, quant, time_elapsed, bound