import numpy as np
import time
from config import get_config

import cvxgraphalgs as cvxgr
import src.offshelf.maxcut.maxcut as maxcut_pkg

from src.ising_gt import ising_ground_truth
from src.util.plottings import laplacian_to_graph



# https://github.com/hermish/cvx-graph-algorithms

def off_the_shelf(cf, laplacian, method):
    graph = laplacian_to_graph(laplacian)
    if method == "random_cut":
        # random cut
        start_time = time.time()
        random_cut = cvxgr.algorithms.random_cut(graph, 0.5)
        end_time = time.time()
        cut_size = random_cut.evaluate_cut_size(graph)
        print('Random Cut Performance')
    elif method == "greedy_cut":
        # greedy cut
        start_time = time.time()
        greedy_cut = cvxgr.algorithms.greedy_max_cut(graph)
        end_time = time.time()
        cut_size = greedy_cut.evaluate_cut_size(graph)
        print('Greedy Cut Performance')
    elif method == "goemans_williamson":
        # Goemans-Williamson Algorithm
        start_time = time.time()
        sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(graph)
        end_time = time.time()
        cut_size = sdp_cut.evaluate_cut_size(graph)
        print('Goemans-Williamson Performance')
    elif method == "sdp_BM":
        # https://github.com/pandrey-fr/maxcut
        # sdp = maxcut_pkg.MaxCutSDP(graph)
        # start_time = time.time()
        # cut_size = sdp.solve(laplacian)
        # end_time = time.time()
        # print("sdp",cut_size)
        sdp = maxcut_pkg.MaxCutBM(graph, dim_p=20)
        start_time = time.time()
        cut_size = sdp.solve(laplacian)
        end_time = time.time()
        print('Burer-Monteiro Performance')
    time_elapsed = end_time - start_time
    # print("Cut size: {}, Time elapsed {:.2f}".format(cut_size, time_elapsed))

    return method+str(cf.input_size), cut_size, time_elapsed