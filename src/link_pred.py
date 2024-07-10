# external imports
from itertools import chain
from networkx import Graph, adjacency_matrix, connected_components, empty_graph, from_edgelist, is_connected, number_of_edges, number_of_nodes
import numpy as np
from numpy.random import Generator, default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# internal imports
from cite.node2vec import train_node2vec
from random_walks import RW_Iterable



def edge_to_idx(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    '''returns custom index (array) for given [edge] (array), dot-multiplies it w/ defined (2,)-array using [n_nodes] so as to map each edge in [edges] to unique int-index'''
    return np.dot(edges, np.array([n_nodes, 1]))  # length: len(edges)


"""
def idx_to_edge(idx: np.ndarray, n_nodes: int) -> np.ndarray:
    '''inversion of edge_to_idx(): returns edge array back from given 1D-[idx]'''
    return np.array(np.divmod(idx, n_nodes)).T  # transposes returned edges back to shape (len(edges), 2)
"""


def hadamard(X: np.ndarray, idx: np.ndarray, n_nodes: int) -> np.ndarray:
    '''returns 3D-array of elem.wise product of slices of [X] according to start- & end-nodes infered from given edge [idx]'''
    edges = np.array(np.divmod(idx, n_nodes))
    return X[edges[0]] * X[edges[1]]
    # decomment below to use idx_to_edge() instead
    #edges = idx_to_edge(idx, n_nodes).T
    #return X[edges[: , 0]] * X[edges[: , 1]]  # for re-transposed edges, see idx_to_edge()




def min_spanning_tree(rng: Generator, adj_mats_connec_comps: list[tuple[np.ndarray, int]]) -> np.ndarray:
    '''algebraic method for min. spanning tree generation using adj.mat.s, returns list of minimum spanning trees (repres. as arrays of edges), one for each connec. comp. implied by [adj_mats_connec_comps]'''
    span_tree_list = []  # collect edge sets of min. span trees over each connec. comp.

    for adj_mat, min_node in adj_mats_connec_comps:
        n_nodes_comp = len(adj_mat)  # number of nodes in current connec. comp.
        span_tree_nodes = np.zeros(n_nodes_comp)  # tree index: array collecting nodes for spanning tree of connec. comp.
        start = rng.choice(n_nodes_comp, size=None, replace=True, p=None, axis=0, shuffle=True)  # start node of spanning tree
        span_tree_nodes[start] = 1  # assign start node to tree index
        span_tree_edges = []  # collect edges for spanning tree

        while(np.sum(span_tree_nodes) < n_nodes_comp):  # while current tree contains less than all nodes in connec. comp.
            frontier = (1 - span_tree_nodes) * np.max(adj_mat * span_tree_nodes, axis=-1)  # indexing array of all potential next nodes
            # uniformly sample next node from all potential next nodes
            next = rng.choice(n_nodes_comp, size=None, replace=True, p = frontier / np.sum(frontier), axis=0, shuffle=True)
            heads = adj_mat[next] * span_tree_nodes  # indexing array of head nodes from which the [next] node *may have been* drawn
            # back-track from next to head node: uniformly sample actual head node from [heads]
            head = rng.choice(n_nodes_comp, size=None, replace=True, p = heads / np.sum(heads), axis=0, shuffle=True)
            span_tree_nodes[next] = 1  # update tree index w/ [next] node
            next_edge = [head, next]
            span_tree_edges.append(next_edge)  # add new edge between head and next node to tree edges

        span_tree_list.append(np.array(span_tree_edges) + min_node)  # add span tree edge array to list w/ re-shifted node indices

    return np.concatenate(span_tree_list, axis=0)



def edge_sampler(rng: Generator, X: np.ndarray,
                 edges_connec_comps: list[np.ndarray], edges_connec_comps_idx: np.ndarray,
                 edges_minus_trees_idx: np.ndarray, all_false_idx: np.ndarray,
                 n_nodes: int, n_edges: int, eval_share: float, sample_strat) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''return data for downstream classifier from randomly sampled edge sets as instructed in sheet; takes in common RNG, trained node2vec embedding X, ("cleaned up", see note below) list of edges in each connec. comp. of the graph, index of all potential false edges, number of nodes, and approx., relative share of (valid) edges for evaluation'''
    train_share = 1 - eval_share
    ## true edges: use edge sampling strategy (set in link_pred())
    if sample_strat == 1:  # min. span. trees
        # index of true edges for eval.:
        # sample ca. [eval_share * n_edges] edges from complement of all (cleaned) edges w.r.t. combined min. span-trees,
        # ensures connectivity of subgraph of train edges
        true_idx_eval = rng.choice(edges_minus_trees_idx, size=int(np.ceil(eval_share * n_edges)), replace=False, p=None, axis=0, shuffle=False)
        # true edges for training = complement of [true_edges_eval] w.r.t. all (cleaned) edges
        true_idx_train = np.setdiff1d(edges_connec_comps_idx, true_idx_eval, assume_unique=True)
    elif sample_strat == 2:  # re-sampling
        true_idx_eval, true_idx_train = [], []  # collect edge indices for each connec. comp.
        for comp_edges in edges_connec_comps:  # run over edge sets per connec. comp.
            comp_is_connected = False
            while not comp_is_connected:
                comp_edges_train = rng.choice(comp_edges, size=int(train_share * len(comp_edges)),
                                       replace=False, p=None, axis=0, shuffle=False)  # sample train edges from connec. comp.
                # check if induced subgraph is connected
                comp_is_connected = is_connected(from_edgelist(list(comp_edges_train)))
            true_idx_train.append(edge_to_idx(comp_edges_train, n_nodes))  # proceed w/ connected train subgraph, add resp. edge index
        # concatenate connec. comp. indices into their resp. indices for eval & train
        true_idx_train = np.concatenate(true_idx_train, axis=0)
        true_idx_eval = np.setdiff1d(edges_connec_comps_idx, true_idx_train, assume_unique=True)
    else:  # generic
        # index of true edges for eval.: ca. [eval_share] of uniformly sampled edges within each (valid) connec. comp., at least 1 per comp.
        true_idx_eval = edge_to_idx(np.concatenate(
            [rng.choice(comp_edges, size=int(np.ceil(eval_share * len(comp_edges))), replace=False, p=None, axis=0, shuffle=False)
                for comp_edges in edges_connec_comps],
            axis=0), n_nodes)
        # true edges for training = complement of [true_edges_eval] w.r.t. all (cleaned up, concatenated) edges
        true_idx_train = np.setdiff1d(edges_connec_comps_idx, true_idx_eval, assume_unique=True)

    ## false edges:
    len_true_eval, len_true_train = len(true_idx_eval), len(true_idx_train)
    # uniformly sample indices w.r.t. all true edges from [all_false_idx]
    false_idx = rng.choice(all_false_idx, size=len_true_eval + len_true_train, replace=False, p=None, axis=0, shuffle=False)
    # split [false_idx] into disjoint eval & train part w/ equal length to resp. true edge parts
    false_idx_eval, false_idx_train = false_idx[: len_true_eval], false_idx[len_true_eval :]

    ## create eval & train data:
    # create edge-lvl input tensors from X w.r.t. eval & train edges, using resp. concatenations of true & false edges
    XX_eval = hadamard(X, np.concatenate([true_idx_eval, false_idx_eval], axis=0), n_nodes)
    XX_train = hadamard(X, np.concatenate([true_idx_train, false_idx_train], axis=0), n_nodes)
    # create binary (+/-1) edge labels w.r.t. true/false edges of eval & train data
    y_eval = np.concatenate([np.ones(len_true_eval), -np.ones(len_true_eval)], axis=0)
    y_train = np.concatenate([np.ones(len_true_train), -np.ones(len_true_train)], axis=0)

    return XX_eval, XX_train, y_eval, y_train  # return inputs XX & labels y for eval & train edges



def link_pred(graph: Graph, p: float, q: float, l: int, l_ns: int, dim: int,  # main parameters, see sheet
               n_batches: int, batch_size: int, device: str, return_train_score: bool,  # extra parameters
               k: int = 5, eval_share: float = 0.2, lr: float = 0.01, set_node_labels: bool = False):  # default settings
    '''runs link prediction (Ex.4) for given [graph] (here: Facebook & PPI): creates custom train & eval edge labels from randomly sampled edge sets as instructed in sheet, trains node2vec embedding [X] first from given parameters, then uses Hadamard products thereof as edge-lvl input to learn & classify custom edge labels by running [k] resampled rounds of logistic regression, returns mean & standard deviation of both accuracy & ROC_AUC scores on eval (& train) edge sets'''
    print("Prepare dataset") if print_progress else print(end="")
    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset of pq-walks
    n_nodes = dataset.n_nodes
    edges = dataset.edges
    rng = default_rng(seed=None)

    # overall false edges:
    print("Compute potential false edges") if print_progress else print(end="")
    # take all unique edge indices in undirected, *complete* graph,
    complete_idx = np.array(list(chain(* [[n_nodes * start_node + end_node
                                           for end_node in range(start_node, n_nodes)]
                                          for start_node in range(n_nodes)])))
    # then subtract indices of full, *directed* edges of [graph], i.e. [edges] concat. w/ flipped copy of itself
    all_false_idx = np.setdiff1d(complete_idx,
                                 edge_to_idx(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), n_nodes),
                                 assume_unique=True)

    # prepare input for classifier
    print("Train node2vec embedding") if print_progress else print(end="")
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, lr)

    # construct classifier object: logistic regression
    print("Construct classifier") if print_progress else print(end="")
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, l1_ratio=None)

    print("Address connected components") if print_progress else print(end="")
    # FIXME In (Citeseer, Cora &) PPI there are connected components which each contain less than 2 edges, including some which only consist of a single node with a self-loop. Since it does not seem meaningful to sample edges for E_eval or E_train from these - in accordance w/ the condition that both edge sets should remain connected within each connected component (see sheet) - we here grant the option to remove them from the edge-lvl data (i.e. "clean" the [graph]) for the downstream classifier. (De)comment the following line marked by '###' accordingly:
    connec_comps = [graph.subgraph(comp).copy() for comp in connected_components(graph)]
    connec_comps = [comp for comp in connec_comps if number_of_edges(comp) >= 2]  ### remove invalid connec. comp.s
    # list of edges in each connected component of (cleaned) graph, subtract 1 elem.wise to account for node count starting at zero, list[(x,2)-np.ndarray]
    edges_connec_comps = [np.array([[edge[0], edge[1]]
                            for edge in comp.edges(data=True)]) - 1
                          for comp in connec_comps]
    # edge index of concatenated (cleaned) connec. comp.s
    edges_connec_comps_idx = edge_to_idx(np.concatenate(edges_connec_comps, axis=0), n_nodes)
    # list of adj.mat.s for each connec. comp., tupled w/ minimum node index in each connec. comp.
    adj_mats_connec_comps = [(adjacency_matrix(comp).toarray(),
                              np.min([node[0] for node in comp.nodes(data=True)]) - 1)
                             for comp in connec_comps]
    """
    choose edge sampling strategies accordingly:
    0/else - generic:       naively samples ca. [eval_share] of edges in each connec. comp.
                            (usually sufficient, fast, termination guaranteed, connectivity likely)
    1 - min. span. trees:   ensures that train edges contain random minimum spanning trees, then samples [eval_share] from rest edges
                            (slow, connectivity & termination guaranteed)
    2 - re-sampling:        same as 'generic', but also re-samples until train edges form a connected subgraph in each connec. comp.
                            (usually fast, connectivity guaranteed, termination not, unreliable for PPI)
    """
    sample_strat = 0

    if sample_strat == 1:
        print("Generate mimimum spanning trees") if print_progress else print(end="")
        trees_combined = min_spanning_tree(rng, adj_mats_connec_comps)  # edge set of combined, random, minimum spanning trees
        #len_trees_combined = len(trees_combined)  # number of edges over all trees
        # subtract combined tree edges from all (cleaned) edges
        edges_minus_trees_idx = np.setdiff1d(edges_connec_comps_idx,
                                         edge_to_idx(trees_combined, n_nodes),
                                         assume_unique=True)
    else:
        edges_minus_trees_idx = np.zeros(0)

    print("Run evaluation loop") if print_progress else print(end="")
    scores_mat = []  # collect lists of scores for each round
    for round in range(k):
        print(f"\tRound {round}") if print_progress else print(end="")
        scores_list = []  # collect scores for current round
        # compute data for classifier
        print("\t\tEdge sampling") if print_progress else print(end="")
        XX_eval, XX_train, y_eval, y_train = edge_sampler(rng, X,
                                                          edges_connec_comps, edges_connec_comps_idx, edges_minus_trees_idx, all_false_idx, n_nodes, dataset.n_edges, eval_share, sample_strat)

        # apply newly labeled edge-lvl data to downstream classifier
        print("\t\tRun classifier") if print_progress else print(end="")
        model = classifier.fit(XX_train, y_train)
        accuracy_eval = accuracy_score(y_eval, model.predict(XX_eval))
        roc_auc_eval = roc_auc_score(y_eval, model.decision_function(XX_eval))
        scores_list = [accuracy_eval, roc_auc_eval]
        if return_train_score:  # also evaluate & return scores on train data
            accuracy_train = accuracy_score(y_train, model.predict(XX_train))
            roc_auc_train = roc_auc_score(y_train, model.decision_function(XX_train))
            scores_list += [accuracy_train, roc_auc_train]
        scores_mat.append(scores_list)

    print("Compute results") if print_progress else print(end="")
    scores_mat = np.array(scores_mat)
    modes = (['train_', 'test_'] if return_train_score else ['test_'])  # choose strings for data [modes]
    score_types = ['accuracy', 'roc_auc']  #'roc_auc_ovr'
    modes_types = list(chain(* [[mode + score_type for score_type in score_types] for mode in modes]))  # string list: data [mode] x [score_type]
    scores_mean, scores_std = np.mean(scores_mat, axis=-1), np.std(scores_mat, axis=-1)  # compute mean & std arrays
    # return mean & standard deviation of the scores of each score type over the [k] eval splices, optionally also over the train splices
    return [(modes_types[i], scores_mean[i], scores_std[i]) for i in range(len(modes_types))]



print_progress = False
if __name__ == "__main__":
    # test link prediction
    import pickle
    import torch as th

    print_progress = True
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 100 #10 #100 #500 #1000 #2000
    n_batches = 100 #10 #100
    dim = 128 #10 #20 #50 #128
    p = 1.0
    q = 1.0
    l = 5
    l_ns = 5
    return_train_score = True

    with open('datasets/Facebook/data.pkl', 'rb') as data:
    #with open('datasets/PPI/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    results = link_pred(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
    print(f"\nMean \u00b1 StD of Scores, rounded:")
    for mode, mean, std in results:
        print(f"{mode}:\t{round(mean * 100, 2)} \u00b1 {round(std * 100, 2)}")
