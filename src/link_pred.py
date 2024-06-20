# external imports
from itertools import chain
from networkx import Graph, connected_components
import numpy as np
from numpy.random import Generator, default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
#from sklearn.model_selection import cross_validate#, cross_val_score

# internal imports
from node2vec import train_node2vec
from random_walks import RW_Iterable



def idx_map(edges: np.ndarray, n_nodes: int) -> list[int]:
    '''returns custom index array for given edge array, dot-multiplies it w/ defined (2,)-array using [n_nodes] so as to map each edge in [edges] to unique int-index'''
    #return list(np.dot(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), np.array([n_nodes, 1])))  # size: 2*len(edges), doubled
    return list(np.dot(edges, np.array([n_nodes, 1])))  # size: len(edges)



def edge_sampler(rng: Generator, edges_connec_comp_cleaned: list[np.ndarray], edges: np.ndarray, n_nodes: int, eval_share: float
                 ) -> tuple[list[int], list[int], list[int], list[int]]:
    '''returns custom edge index lists from randomly sampled edge sets as instructed in sheet; uses common RNG, full edge set of an assoc. graph, ("cleaned up", see note below) list of edges in each connec. comp. of the graph, number of nodes & approx., rel. share of (valid) edges for evaluation'''
    # positive edge samples for evaluation: ca. [eval_share] of edges within each (valid) connec. comp. uniformly sampled (at least 1), concatenated & index-mapped
    pos_samples_eval = idx_map(np.concatenate(
        [rng.choice(comp_edges, size=int(np.ceil(eval_share * len(comp_edges))), replace=False, p=None, axis=0, shuffle=False)
         for comp_edges in edges_connec_comp_cleaned], axis=0), n_nodes)

    # positive edge samples for training = complement of edges_eval_idx w.r.t. all (cleaned up) edges (concatenated)
    pos_samples_train = list(set(idx_map(np.concatenate(edges_connec_comp_cleaned, axis=0), n_nodes)).difference(set(pos_samples_eval)))

    # overall negative edges: take flattened set of all unique edge indices in undirected, loopless, *complete* graph
    neg_samples = list(set(chain(* [[node_start * n_nodes + node_end
                                     for node_end in range(node_start + 1, n_nodes)]
                                    for node_start in range(n_nodes)])
    # then subtract accordingly structured indices of full, *directed* edge set of [graph], i.e. [edges] concat. w/ flipped copy of itself
                           ).difference(set(idx_map(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), n_nodes))))

    # uniformly sample negative samples for resp. edge index maps from overall negative edges
    neg_samples_eval = list(rng.choice(neg_samples, size=len(pos_samples_eval), replace=False, p=None, axis=0, shuffle=False))
    neg_samples_train = list(rng.choice(neg_samples, size=len(pos_samples_train), replace=False, p=None, axis=0, shuffle=False))

    return pos_samples_eval, neg_samples_eval, pos_samples_train, neg_samples_train  # return positive & negative edge index samples from eval & train edge sets



def link_pred(graph: Graph, p: float, q: float, l: int, l_ns: int, dim: int,  # main parameters, see sheet
               n_batches: int, batch_size: int, device: str, return_train_score: bool,  # extra parameters
               k: int = 5, set_node_labels: bool = False, eval_share: float = 0.2):# -> tuple[np.float64, np.float64]:  # default settings
    '''runs link prediction (Ex.4) for given [graph] (here: Facebook & PPI): creates custom edge labels from randomly sampled edge sets as instructed in sheet, trains node2vec embedding X first from given parameters, then uses its Hadamard products XX as edge-lvl input to learn & classify custom edge labels by running [k] resampled rounds of logistic regression, returns mean & std of both accuracy & ROC_AUC scores on evaluation edge sets'''
    print("Prepare dataset")
    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset of pq-walks
    n_nodes = dataset.n_nodes
    n_nodes_squared = n_nodes ** 2
    #n_edges = dataset.n_edges
    #edges = dataset.edges
    #rng = dataset.rng
    rng = default_rng(seed=None)

    # prepare input for classifier
    print("Compute Hadamard edge tensor XX")
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device)  # train node2vec embedding
    # use Hadamard products of X between each node pair as edge-lvl inputs to classifier
    # FIXME RAM spills over into hard drive for too large [dim] (even for >=128 for me), slows down dramatically, possible slu. is slated
    """
    XX = np.array(list(chain(* [[np.multiply(X[node_start], X[node_end])
                                 for node_end in range(n_nodes)]
                                for node_start in range(n_nodes)])))  # flattened 2D-np.ndarray, n_nodes**2 x dim
    """
    XX = np.concatenate([np.kron(X[: , i], X[: , i]).reshape((n_nodes_squared, 1)) for i in range(dim)], axis=-1)

    # construct classifier object: logistic regression
    print("Construct classifier")
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, l1_ratio=None) #multi_class='auto'

    print("Address connected components")
    # list of edges in each connected component of graph, subtract 1 elem.wise to account for node count starting at zero, list[2D-np.ndarray]
    edges_connec_comp = [np.array([[edge[0], edge[1]]
                                   for edge in graph.subgraph(comp).copy().edges(data=True)]) - 1
                         for comp in connected_components(graph)]
    edges_connec_comp_cleaned = edges_connec_comp

    # FIXME In (Citeseer, Cora &) PPI there are connected components which each contain less than 2 edges, including some which only consist of a single node with a self-loop. Since it does not seem meaningful to sample edges for E_eval or E_train from these - in accordance w/ the condition that both edge sets should remain connected within each connected component (see sheet) - we here grant the option to remove them from the edge-lvl data for the downstream classifier. (De)comment the following for-block accordingly:
    """
    for comp_edges in edges_connec_comp:
        if len(comp_edges) < 2:  # given a connected component w/ less than 2 edges
            edges_connec_comp_cleaned.remove(comp_edges)  # remove it from the list of connec.comp.edges  # FIXME error for PPI
    """
    # proceed w/ edges_connec_comp_cleaned...

    print("Run evaluation loop")
    y = np.ones(n_nodes_squared)
    scores_mat = []
    for round in range(k):
        print(f"\tround {round}")
        scores_list = []
        # positive & negative edge index samples from eval & train edge sets
        print("\t\tedge sampling")
        pos_samples_eval, neg_samples_eval, pos_samples_train, neg_samples_train = edge_sampler(rng, edges_connec_comp_cleaned, dataset.edges, n_nodes, eval_share)
        # concatenate resp. positive & negative edge index samples
        samples_eval = np.concatenate([pos_samples_eval, neg_samples_eval], axis=0)
        samples_train = np.concatenate([pos_samples_train, neg_samples_train], axis=0)
        # create training & evaluation edge labels (binary: +/-1) by concatenating slices w.r.t. resp. positive & negative edge index samples
        y_eval = np.concatenate([y[pos_samples_eval], -y[neg_samples_eval]], axis=0)
        y_train = np.concatenate([y[pos_samples_train], -y[neg_samples_train]], axis=0)
        XX_eval, XX_train = XX[samples_eval], XX[samples_train]  # training & evaluation slices from edge-lvl input XX

        # apply newly labeled edge-lvl data to downstream classifier
        print("\t\trun classifier")
        model = classifier.fit(XX_train, y_train)#.predict(XX_eval)
        accuracy_eval = accuracy_score(y_eval, model.predict(XX_eval))
        roc_auc_eval = roc_auc_score(y_eval, model.decision_function(XX_eval))
        scores_list = [accuracy_eval, roc_auc_eval]
        if return_train_score:
            accuracy_train = accuracy_score(y_train, model.predict(XX_train))
            roc_auc_train = roc_auc_score(y_train, model.decision_function(XX_train))
            scores_list += [accuracy_train, roc_auc_train]
        scores_mat.append(scores_list)

    print("Compute results")
    scores_mat = np.array(scores_mat)
    modes = (['train_', 'test_'] if return_train_score else ['test_'])  # choose strings for data [modes]
    score_types = ['accuracy', 'roc_auc']  #'roc_auc_ovr'
    modes_types = list(chain(* [[mode + score_type for score_type in score_types] for mode in modes]))  # string list: data [mode] x [score_type]
    scores_mean, scores_std = np.mean(scores_mat, axis=-1), np.std(scores_mat, axis=-1)  # compute mean & std arrays
    # return mean & standard deviation of the scores of each score type over the [k] eval splices, optionally also over the train splices
    return [(modes_types[i], scores_mean[i], scores_std[i]) for i in range(len(modes_types))]



if __name__ == "__main__":
    # test link prediction
    import pickle
    import torch as th

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 10 #10 #100
    n_batches = 100 #10 #100
    dim = 50 #10 #20 #128
    p = 1.0
    q = 1.0
    l = 5
    l_ns = 5
    set_node_labels = False
    return_train_score = True

    with open('datasets/Facebook/data.pkl', 'rb') as data:
    #with open('datasets/PPI/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    results = link_pred(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
    #print(mean.dtype, std.dtype)
    print(f"\nMean \u00b1 StD of Scores, rounded in %:")
    for mode, mean, std in results:
        print(f"{mode}:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")
