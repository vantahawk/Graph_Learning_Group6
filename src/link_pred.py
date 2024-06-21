# external imports
from itertools import chain
from networkx import Graph, connected_components
import numpy as np
from numpy.random import Generator, default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# internal imports
from node2vec import train_node2vec
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
    return np.multiply(X[edges[0]], X[edges[1]])
    # decomment below to use idx_to_edge() instead
    #edges = idx_to_edge(idx, n_nodes).T
    #return np.array([np.multiply(X[edges[: , 0]], X[edges[: , 1]]) for edge in idx])  # for re-transposed edges, see idx_to_edge()



def edge_sampler(rng: Generator, X: np.ndarray, edges_connec_comp_cleaned: list[np.ndarray], all_false_idx: np.ndarray,
                 n_nodes: int, eval_share: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''return data for downstream classifier from randomly sampled edge sets as instructed in sheet; takes in common RNG, trained node2vec embedding X, ("cleaned up", see note below) list of edges in each connec. comp. of the graph, index of all potential false edges, number of nodes, and approx., relative share of (valid) edges for evaluation'''
    ## true edges:
    # index of true edges for evaluation: ca. [eval_share] of uniformly sampled edges within each (valid) connec. comp., at least 1 per comp.
    true_idx_eval = edge_to_idx(np.concatenate(
        [rng.choice(comp_edges, size=int(np.ceil(eval_share * len(comp_edges))), replace=False, p=None, axis=0, shuffle=False)
         for comp_edges in edges_connec_comp_cleaned], axis=0), n_nodes)
    # TODO replace w/ Benedict's spanning tree method

    # true edges for training = complement of [true_edges_eval] w.r.t. all (cleaned up, concatenated) edges
    true_idx_train = np.setdiff1d(edge_to_idx(np.concatenate(edges_connec_comp_cleaned, axis=0), n_nodes), true_idx_eval, assume_unique=True)

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
    #
    # take all unique edge indices in undirected, *complete* graph,
    complete_idx = np.array(list(chain(* [[n_nodes * start_node + end_node
                                           for end_node in range(start_node, n_nodes)]
                                          for start_node in range(n_nodes)])))
    # then subtract indices of full, *directed* edges of [graph], i.e. [edges] concat. w/ flipped copy of itself
    all_false_idx = np.setdiff1d(complete_idx, edge_to_idx(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), n_nodes), assume_unique=True)
    #

    # prepare input for classifier
    print("Train node2vec embedding") if print_progress else print(end="")
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, lr)

    # construct classifier object: logistic regression
    print("Construct classifier") if print_progress else print(end="")
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, l1_ratio=None)

    print("Address connected components") if print_progress else print(end="")
    # list of edges in each connected component of graph, subtract 1 elem.wise to account for node count starting at zero, list[(x,2)-np.ndarray]
    edges_connec_comp = [np.array([[edge[0], edge[1]]
                                   for edge in graph.subgraph(comp).copy().edges(data=True)]) - 1
                         for comp in connected_components(graph)]
    edges_connec_comp_cleaned = edges_connec_comp

    # FIXME In (Citeseer, Cora &) PPI there are connected components which each contain less than 2 edges, including some which only consist of a single node with a self-loop. Since it does not seem meaningful to sample edges for E_eval or E_train from these - in accordance w/ the condition that both edge sets should remain connected within each connected component (see sheet) - we here grant the option to remove them from the edge-lvl data for the downstream classifier. (De)comment the following line accordingly:
    edges_connec_comp_cleaned = [comp_edges for comp_edges in edges_connec_comp if len(comp_edges) >= 2]  # remove invalid connec. comp.s
    # proceed w/ edges_connec_comp_cleaned...

    print("Run evaluation loop") if print_progress else print(end="")
    scores_mat = []  # collect lists of scores for each round
    for round in range(k):
        print(f"\tRound {round}") if print_progress else print(end="")
        scores_list = []  # collect scores for current round
        # compute data for classifier
        print("\t\tEdge sampling") if print_progress else print(end="")
        XX_eval, XX_train, y_eval, y_train = edge_sampler(rng, X, edges_connec_comp_cleaned, all_false_idx, n_nodes, eval_share)

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
