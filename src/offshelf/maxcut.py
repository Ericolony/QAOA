
##############################################################################
########################### ''' GW algorithm ''' #############################
##############################################################################

# Copyright 2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Goemans-Williamson classical algorithm for MaxCut
"""

from typing import Tuple

import cvxpy as cvx
import networkx as nx
import numpy as np
import cvxopt

def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """
    # Kudos: Originally implementation by Nick Rubin, with refactoring and
    # cleanup by Jonathon Ward and Gavin E. Crooks
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # Setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)
    obj = cvx.Maximize(cvx.trace(laplacian * psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T

    # Bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)

    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector /= np.linalg.norm(random_vector)
    colors = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = colors @ laplacian @ colors.T
    return colors, score, bound
##############################################################################
##############################################################################
##############################################################################

##############################################################################
######################## ''' Trevisan algorithm ''' ##########################
##############################################################################

if __name__ == '__main__':
    # Quick test of GW code
    from src.util.plottings import laplacian_to_graph
    from src.util.helper import load_data
    from config import get_config
    cf, unparsed = get_config()
    laplacian = load_data(cf)
    G = laplacian_to_graph(laplacian)

    laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
    bound = goemans_williamson(G)[2]

    # assert np.isclose(bound, 36.25438489966327)
    print(goemans_williamson(G))

    scores = [goemans_williamson(G)[1] for n in range(128)]
    # assert max(scores) >= 34

    print(min(scores), max(scores))

# https://github.com/rigetti/quantumflow-qaoa/blob/master/gw.py