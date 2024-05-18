import networkx as nx
#import typing
#from networkx import number_of_nodes
import numpy as np
import torch as th
#import argparse



def max_node_dim(graphs: list[nx.Graph]) -> int:
    '''returns max. number of nodes of [graphs] within a given dataset (as list of undirected graphs), used for uniform zero-padding along node-axis'''
    max_n_nodes = 0  # max number of nodes in dataset thus far
    for graph in graphs:
        n_nodes = nx.number_of_nodes(graph)  # current graph
        if n_nodes > max_n_nodes:  # assign if larger
            max_n_nodes = n_nodes
    return max_n_nodes



def norm_adjacency(graph: nx.Graph) -> np.ndarray:
    '''returns degree-normalized adjacency matrix of a given undirected [graph]'''
    A = nx.adjacency_matrix(graph).toarray()
    N = nx.number_of_nodes(graph)
    A_norm = np.empty((N, N), dtype=float)  # initialize normed adj.mat.

    # assign elements of normed adj.mat.
    for i in range(N):
        A_norm[i][i] = 1 / (np.sum(A[:, i]) + 1)  # simplified for i=j
        for j in range(i+1, N):  # run over upper-right triangle of A
            if A[i][j] == 0:  # if no edge
                A_norm[i][j] = 0
            else:
                A_norm[i][j] = 1 / np.sqrt((np.sum(A[i]) + 1) * (np.sum(A[j]) + 1))  # see exercise sheet
            A_norm[j][i] = A_norm[i][j]  # due to symmetry of A

    return A_norm  # not zero-padded yet



def zero_pad(array: np.ndarray, axes: list[int], length: int) -> np.ndarray:
    '''returns [array] zero-padded up to [length] for a given list of [axes]'''
    array_shape = array.shape
    pad_rule = []  # list of tuples to configure zero-padding by pad()
    for axis in range(len(array_shape)):
        if axis in axes:
            pad_rule.append((0, max(0, length - array_shape[axis])))  # extra zeros on axis up to length
        else:
            pad_rule.append((0, 0))  # no extra zeros
    return np.pad(array, pad_rule)



def stack_adjacency(graphs: list[nx.Graph], length: int) -> th.Tensor:
    '''produce norm.ed adj.mat.s of every graph in [graphs], zero-pad them up to [length], and stack along batch axis'''
    return th.tensor(np.array([zero_pad(norm_adjacency(graph) ,[0,1], length) for graph in graphs]))


#def zero_pad_node_lvl(graph_array: np.ndarray, length: int) -> np.ndarray:
#    return zero_pad(graph_array, [0], length)


#if __name__ == "__main__":  # optional: Ex.1 demo
