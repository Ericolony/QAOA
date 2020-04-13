#!/usr/bin/env python

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


def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
	"""
	The Goemans-Williamson algorithm for solving the maxcut problem.
	Ref:
		Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
		algorithms for maximum cut and satisfiability problems using
		semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
	Returns:
		np.ndarray: Graph coloring (+/-1 for each node)
		float:	  The GW score for this cut.
		float:	  The GW bound from the SDP relaxation
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



def maxcut_random(G, is_adj_list=True):
	# calculates the maximum weight cut by randomly assigning vertices to a cut. This is
	# a 1/2-approximation algorithm in expectation.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	if len(G) < 1:
		return None
	if len(G) == 1:
		return [1]
	# return [1 if np.random.sample() >= 0.5 else -1 for i in range(len(G))]
	chi = np.zeros(len(G))
	resevoir = [1 if elem < len(G) // 2 else -1 for elem in range(len(G))]
	for i in range((len(G))):
		chi[i] = resevoir.pop(np.random.randint(0, len(resevoir)))
	return chi

def maxcut_greedy(G, is_adj_list=True):
	# calculates the maximum weight cut using a simple greedy implementation. This is a
	# 1/2-approximation algorithm.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	
	l, r, V, l_cost, r_cost = set(), set(), len(G), 0, 0
	if is_adj_list:
		for num, v in enumerate(G):
			for u in v:
				if isinstance(u, tuple):
					if u[0] in l:
						l_cost += u[1]
					if u[0] in r:
						r_cost += u[1]
				else:
					if u in l:
						l_cost += 1
					if u in r:
						r_cost += 1
			if l_cost <= r_cost: # r_cost is amount by which cut will increase
				l.add(num)
			else:
				r.add(num)
			l_cost, r_cost = 0, 0
	else:
		for i in range(V):
			for j in range(V):
				if j in l:
					l_cost += G[i, j]
				if j in r:
					r_cost += G[i, j]
			if l_cost <= r_cost:
				l.add(i)
			else:
				r.add(i)
			l_cost, r_cost = 0, 0
	return [1 if i in l else -1 for i in range(V)]


def maxcut_SDP(G, solver, is_adj_list=True):
	# calculates the maximum weight cut by generating |V| vectors with a vector program,
	# then generating a random plane that cuts the vertices. This is a .878-approximation
	# algorithm.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	# setup
	V, constraints, expr = len(G), [], 0
	if is_adj_list:
		G = _adj_list_to_adj_matrx(G)

	# variables				
	X = cvx.Variable((V, V), PSD=True)

	# constraints
	for i in range(V):
		constraints.append(X[i, i] == 1)

	# objective function	
	expr = cvx.sum(cvx.multiply(G, (np.ones((V, V)) - X)))

	# solve
	prob = cvx.Problem(cvx.Maximize(expr), constraints)

	# ['CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCS']
	prob.solve(solver=solver)

	# random hyperplane partitions vertices
	Xnew = X.value
	eigs = np.linalg.eigh(Xnew)[0]
	if min(eigs) < 0:
		Xnew = Xnew + (1.00001 * abs(min(eigs)) * np.identity(V))
	elif min(eigs) == 0:
		Xnew = Xnew + 0.0000001 * np.identity(V)
	x = np.linalg.cholesky(Xnew).T
	
	r = np.random.normal(size=(V))
	return [1 if np.dot(x[:,i], r) >= 0 else -1 for i in range(V)]


def debug2(G):
	V = len(G)
	X = cvx.Variable((V, V), PSD=True)

	constraints = [X >> 0]
	for i in range(V):
		constraints.append(X[i, i] == 1)

	diag = np.diag(G.sum(axis=-1))
	L = -0.25*(diag - G)

	expr = cvx.trace(L@X)
	prob = cvx.Problem(cvx.Minimize(expr), constraints)
	prob.solve()

	Xnew = X.value
	eigs = np.linalg.eigh(Xnew)[0]
	if min(eigs) < 0:
		Xnew = Xnew + (1.00001 * abs(min(eigs)) * np.identity(V))
	elif min(eigs) == 0:
		Xnew = Xnew + 0.0000001 * np.identity(V)
	x = np.linalg.cholesky(Xnew)
	
	assignment = x
	partition = np.random.normal(size=V)
	projections = assignment @ partition
	sides = np.sign(projections)
	return sides


def debug(adjacency):
	size = len(adjacency)
	ones_matrix = np.ones((size, size))
	products = cvx.Variable((size, size), PSD=True)
	cut_size = 0.5 * cvx.sum(cvx.multiply(adjacency, ones_matrix - products))

	objective = cvx.Maximize(cut_size)
	constraints = [cvx.diag(products) == 1]
	problem = cvx.Problem(objective, constraints)
	problem.solve()

	eigenvalues, eigenvectors = np.linalg.eigh(products.value)
	eigenvalues = np.maximum(eigenvalues, 0)
	diagonal_root = np.diag(np.sqrt(eigenvalues))
	assignment = diagonal_root @ eigenvectors.T

	partition = np.random.normal(size=size)
	projections = assignment.T @ partition
	sides = np.sign(projections)
	return sides

def _adj_matrix_to_adj_list(mat):
	# converts an np.ndarray of shape (|V|, |V|) of non-negative real values to a graph 
	# adjacency list. 
	#
	# input:
	#	mat: a list of |V| lists, each containing ints in {0, |V|-1} or 2-tuples of an
	# 	int in {0, |V|-1} and a positive real weight.
	#
	# output:
	#	G: np.ndarray of shape (|V|, |V|)
	return None

def _adj_list_to_adj_matrx(lst):
	# converts an adjacency list np.ndarray of shape (|V|, |V|) of non-negative real 
	# values.
	#
	# input:
	#	lst: np.ndarray of shape (|V|, |V|)
	#
	# output:
	#	G: a list of |V| lists, each containing ints in {0, |V|-1} or 2-tuples of an
	# 	int in {0, |V|-1} and a positive real weight.
	V, weighted = len(lst), False
	for i in range(V):
		if len(lst[i]) > 0:
			if isinstance(lst[i][0], tuple):
				weighted = True
				break
	weighted_adj_matrix = np.zeros((V, V))
	for i in range(V):
		for j in range(len(lst[i])):
			if weighted:
				u, v = i, lst[i][j][1]
				weighted_adj_matrix[u, v] = lst[i][j][0]
				weighted_adj_matrix[v, u] = lst[i][j][0]
			else:
				u, v = i, lst[i][j]
				weighted_adj_matrix[u, v] = 1
				weighted_adj_matrix[v, u] = 1
	return weighted_adj_matrix

def _gnp_random_graph_adj_matrix(n, prob, weighted=False):
	# generates a Gnp random graph: a graph with n vertices where each edge occurs with
	# probability p
	#
	# input:
	#	n: the number of vertices in the graph
	#	p: the probability of each edge existing
	# 
	# output:
	#	A: a numpy array of dimensions n by n representing the adjacency matrix
	A = np.zeros((n, n))
	if not weighted:
		for i in range(n):
			for j in range(i + 1, n):
				sample = np.random.choice((0, 1), p=[1 - prob, prob])
				A[i, j] = sample
				A[j, i] = sample
	else:
		for i in range(n):
			for j in range(i + 1, n):
				if np.random.random_sample() >= 1 - prob:
					sample = max(np.random.normal(1, .5), 0)
					A[i, j] = sample
					A[j, i] = sample
	return A

def _eval_cut(G, chi):
	# calculates total weight across a cut
	#
	# input:
	#	G: a numpy array representing an adjacency matrix
	#	chi: an array where all elements are +1 or -1, representing which side of the cut
	#	that vertex is in.
	#
	#
	total, V = 0, G.shape[0]
	for i in range(V):
		for j in range(i + 1, V):
			if chi[i] != chi[j]:
				total += G[i, j]

	diag = np.diag(G.sum(axis=-1))
	L = 0.25*(diag - G)
	total1 = chi.T@L@chi
	assert(total1==total)
	return total