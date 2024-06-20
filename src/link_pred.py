# external imports
from itertools import chain
from networkx import Graph, connected_components, adjacency_matrix
import numpy as np
from numpy.random import Generator, default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
#from sklearn.model_selection import cross_validate#, cross_val_score

# internal imports
from node2vec import train_node2vec
from random_walks import RW_Iterable
from copy import deepcopy

import argparse


def idx_map(edges: np.ndarray, n_nodes: int) -> list[int]:
    '''returns custom index array for given edge array, dot-multiplies it w/ defined (2,)-array using [n_nodes] so as to map each edge in [edges] to unique int-index'''
    #return list(np.dot(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), np.array([n_nodes, 1])))  # size: 2*len(edges), doubled
    return list(np.dot(edges, np.array([n_nodes, 1])))  # size: len(edges)



def edge_sampler(rng: Generator, edges_connec_comp_cleaned: list[np.ndarray], edges: np.ndarray, n_nodes: int, adj_matrix:np.ndarray, eval_share: float) -> tuple[list[int], list[int], list[int], list[int]]:
    '''returns custom edge index lists from randomly sampled edge sets as instructed in sheet:
    - first find one minimum spanning tree in each connected component, to make sure the train graph components are still connected
    - then sample positive & negative edge index samples for train & eval edge sets
    
    Args:
        rng (Generator): initiliazed RNG
        edges_connec_comp_cleaned (list[np.ndarray]): list of edges in each connected component of graph
        edges (np.ndarray): full edge set of an associated graph
        n_nodes (int): number of nodes
        eval_share (float): approx. relative share of (valid) edges for evaluation
    
    Returns:
        tuple[list[int], list[int], list[int], list[int]]: positive & negative edge index samples for train and eval edge sets'''

    #first find one minimum spanning tree in each connected component
    trees:list[list[int]] = [[rng.choice(comp.flatten())] for comp in edges_connec_comp_cleaned] #sample initial node, in the end the tree should have all the nodes from the connected component
    edge_trees:list[list[np.ndarray]] = [[] for _ in trees] #the edges in the tree
    entire_nodes:list[np.ndarray] = [np.unique(comp.flatten()) for comp in edges_connec_comp_cleaned] #the reference nodes per connected component

    #breadth first search in all connected components to find the minimum spanning trees
    for c, _ in enumerate(edges_connec_comp_cleaned):
        curr_heads = [trees[c][0]] #start with the first head, but make a deepcopy to not run into object reference problems
        #now do a steps from each of the heads until we reached all the nodes
        while len(trees[c]) < len(entire_nodes[c]):
            all_the_new_heads = []
            #shuffle the curr_heads, so its different in each edge_sampler run
            rng.shuffle(curr_heads)
            for head in curr_heads:
                new_heads = np.nonzero(adj_matrix[head])[0] #go to next nodes from the head
                #possible optimization, use the entire_nodes as a negative and switch half-way to speed up the isin search.
                new_heads = new_heads[np.isin(new_heads, trees[c], invert=True, assume_unique=True)]#only keep the ones not already found   

                new_edges = [np.array([head, new]) for new in new_heads] #find the edges to the new heads
                
                new_heads = list(new_heads)
                trees[c].extend(new_heads) #append the new head to the tree, so that for the next head we ignore them
                edge_trees[c].extend(new_edges) #append the new edges to the tree
                all_the_new_heads.extend(new_heads) #append the new heads to the list of heads, for the next round

            curr_heads = all_the_new_heads
        
        #convert into ndarray
        edge_trees[c] = np.array(edge_trees[c])


    def convert2DEdges1D(edges:np.ndarray, n_nodes)->np.ndarray:
        """Converts 2d edges into a 1d representation, by adding the nodes up, after multiplying the first by n_nodes"""
        return np.sum(edges * np.array([n_nodes, 1]), axis=1)

    def convert1DEdges2D(edges:np.ndarray, n_nodes)->np.ndarray:
        """Converts 1d edges into a 2d representation, by dividing the 1d edges by n_nodes and taking the modulo to get the first and second node respectively."""
        return np.array(np.divmod(edges, n_nodes)).T
    
    #now we have the minimum spanning trees, we can sample so that we ignore their edges
    #for that we exclude the tree from the available edges
    for c, comp in enumerate(edges_connec_comp_cleaned):
        #skip empty edge trees, might happen if there are single node components with self loops
        if edge_trees[c].shape[0] == 0:
            continue
        #convert the edges to 1D
        comp1d = convert2DEdges1D(comp, n_nodes)
        tree1d = convert2DEdges1D(edge_trees[c], n_nodes)
        #diff the tree out of the component
        comp1d = np.setdiff1d(comp1d, tree1d)
        #convert back to 2D
        edges_connec_comp_cleaned[c] = convert1DEdges2D(comp1d, n_nodes)


    # positive edge samples for evaluation: ca. [eval_share] of edges within each (valid) connec. comp. uniformly sampled (at least 1), concatenated & index-mapped
    pos_samples_eval = idx_map(np.concatenate(
            [rng.choice(comp_edges, size=int(np.ceil(eval_share * len(comp_edges))), replace=False, p=None, axis=0, shuffle=False)
                for comp_edges in edges_connec_comp_cleaned
            ], 
            axis=0
        ), 
        n_nodes
    )

    # positive edge samples for training = complement of edges_eval_idx w.r.t. all (cleaned up) edges (concatenated)
    pos_samples_train = set(idx_map(np.concatenate(edges_connec_comp_cleaned, axis=0), n_nodes)).difference(pos_samples_eval)

    # overall negative edges: take flattened set of all unique edge indices in undirected, loopless, *complete* graph
    neg_samples = set(
        chain(* [
            [node_start * n_nodes + node_end
                for node_end in range(node_start + 1, n_nodes)
            ]
            for node_start in range(n_nodes)
        ])
    # then subtract accordingly structured indices of full, *directed* edge set of [graph], i.e. [edges] concat. w/ flipped copy of itself
    ).difference(idx_map(np.concatenate([edges, np.flip(edges, axis=-1)], axis=0), n_nodes))

    # uniformly sample negative samples for resp. edge index maps from overall negative edges
    neg_samples_eval = rng.choice(list(neg_samples), size=len(pos_samples_eval), replace=False, p=None, axis=0, shuffle=False)
    neg_samples_train = rng.choice(list(neg_samples.difference(neg_samples_eval)), size=len(pos_samples_train), replace=False, p=None, axis=0, shuffle=False)

    return list(pos_samples_train), list(neg_samples_train), list(pos_samples_eval), list(neg_samples_eval)  # return positive & negative edge index samples from eval & train edge sets



def link_pred(graph: Graph, p: float, q: float, l: int, l_ns: int, dim: int,  # main parameters, see sheet
               n_batches: int, batch_size: int, device: str, return_train_score: bool,  # extra parameters
               k: int = 5, set_node_labels: bool = False, eval_share: float = 0.2):# -> tuple[np.float64, np.float64]:  # default settings
    '''runs link prediction (Ex.4) for given [graph] (here: Facebook & PPI): creates custom edge labels from randomly sampled edge sets as instructed in sheet, trains node2vec embedding X first from given parameters, then uses its Hadamard products XX as edge-lvl input to learn & classify custom edge labels by running [k] resampled rounds of logistic regression.
    
    Args:
        graph (Graph): input graph
        p (float): node2vec parameter
        q (float): node2vec parameter
        l (int): walk length
        l_ns (int): number of walks per node
        dim (int): embedding dimension
        n_batches (int): number of batches for node2vec training
        batch_size (int): batch size for node2vec training
        device (str): device to run on
        return_train_score (bool): whether to return train scores
        k (int): number of resampled rounds
        set_node_labels (bool): whether to set node labels for classification
        eval_share (float): approx. relative share of (valid) edges for evaluation
    
    Returns:
        list[tuple[np.float64, np.float64]]: mean & std of both accuracy & ROC_AUC scores on evaluation edge sets, optionally also on training edge sets
    '''
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
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, lr=0.01)  # train node2vec embedding
    # use Hadamard products of X between each node pair as edge-lvl inputs to classifier

    XX = np.concatenate([np.kron(X[: , i], X[: , i]).reshape((n_nodes_squared, 1)) for i in range(dim)], axis=-1)

    # construct classifier object: logistic regression
    print("Construct classifier")
    classifier = LogisticRegression(
        penalty='l2',
        tol=0.0001, 
        C=1.0, 
        solver='lbfgs', 
        max_iter=100, verbose=0
    )

    print("Address connected components")
    # list of edges in each connected component of graph, subtract 1 elem.wise to account for node count starting at zero, list[2D-np.ndarray]
    edges_connec_comp = [np.array([[edge[0], edge[1]]
                                   for edge in graph.subgraph(comp).copy().edges(data=False)]) - 1
                         for comp in connected_components(graph)]
    edges_connec_comp_cleaned = edges_connec_comp

    print("Run evaluation loop")
    y = np.ones(n_nodes_squared)
    scores_mat = []
    for round in range(k):
        print(f"\tround {round}")
        scores_list = []
        # positive & negative edge index samples from eval & train edge sets
        print("\t\tedge sampling")
        adj_matrix:np.ndarray = adjacency_matrix(graph).toarray()
        edges_connec_comp_copy = deepcopy(edges_connec_comp_cleaned)
        pos_samples_train, neg_samples_train, pos_samples_eval, neg_samples_eval = edge_sampler(rng, edges_connec_comp_copy, dataset.edges, n_nodes, adj_matrix, eval_share)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PPI', help='Dataset to use: \"PPI\" or \"Facebook\"')

    dataset = parser.parse_args().dataset
    if dataset not in ['PPI', 'Facebook']:
        raise ValueError("Dataset must be either \"PPI\" or \"Facebook\"")

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 2000 
    n_batches = 100 
    dim = 128 
    p = 1.0
    q = 1.0
    l = 5
    l_ns = 5
    set_node_labels = False
    return_train_score = True
    #other values are default except lr, which is set in link_pred to 0.01

    print(f"Loading dataset {dataset}")

    with open(f'datasets/{dataset}/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    results = link_pred(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
    #print(mean.dtype, std.dtype)
    print(f"\nMean \u00b1 StD of Scores for dataset {dataset}, rounded in %:")
    for mode, mean, std in results:
        print(f"{mode}:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")
