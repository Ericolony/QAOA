import numpy as np
import time

import cvxgraphalgs as cvxgr
from src.util.plottings import laplacian_to_graph
from src.util.helper import load_data
from config import get_config

cf, unparsed = get_config()
laplacian = load_data(cf)
graph = laplacian_to_graph(laplacian)


# Goemans-Williamson Algorithm
start_time = time.time()
sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(graph)
end_time = time.time()
print('Goemans-Williamson Performance')
print('Cut size:', sdp_cut.evaluate_cut_size(graph))
print("Time ellapsed {}".format(end_time - start_time))

# random cut
start_time = time.time()
random_cut = cvxgr.algorithms.random_cut(graph, 0.5)
end_time = time.time()
print('Random Cut Performance')
print('Cut size:', random_cut.evaluate_cut_size(graph))
print("Time ellapsed {}".format(end_time - start_time))

# greedy cut
start_time = time.time()
greedy_cut = cvxgr.algorithms.greedy_max_cut(graph)
end_time = time.time()
print('Greedy Cut Performance')
print('Cut size:', greedy_cut.evaluate_cut_size(graph))
print("Time ellapsed {}".format(end_time - start_time))



