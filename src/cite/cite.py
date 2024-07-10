'''implementation for running model for Sheet 5/Ex.2 on dataset CITE: produces node2vec-embedding on full graph using random p-trees, then runs k-fold cross-validation on graph- & embedding-splits of rest graph (starting at node 1000), then produces predicted node_labels for first 1000 nodes in CITE'''
# internal imports:
from networkx import Graph, DiGraph, MultiDiGraph
import numpy as np
from numpy.random import default_rng
from os.path import exists
import pickle
from torch import Tensor, float64

# external imports:
from sparse_graph import Sparse_Graph
from random_trees import RT_Iterable
from node2vec import train_node2vec
from closed_walks import CW_kernel
from gnn import run_model



def standard_norm(x: Tensor) -> Tensor:
    '''overall standard score normalization, here for normalizing node2vec embedding tensor'''
    return ((x - x.mean()) / x.std())#.type(float64)



def geom_norm(x: Tensor) -> Tensor:
    '''custom normalization using products along embedding dimension & arithmetic mean over nodes, here for normalizing closed walk kernel embedding tensor'''
    #return x / (x.abs().prod(dim=-1, dtype=float64) ** (1 / x.shape[-1])).mean()
    return x / (x.abs().prod(dim=-1, dtype=float64) ** 0.5).mean()



def log_norm(x: Tensor) -> Tensor:
    '''custom normalization similar to elem.wise base-2-logarithm, here for compressing scale variation of closed walk kernel embedding tensor, optionally shifts x into positive range to avoid one-off non-positive values from approximation (see closed_walks.py)'''
    epsilon = 1e-3
    x_min = x.min()
    if x_min <= 0:
        x = x + x_min + epsilon
    return x.log2() #log() #log10()



def run_evaluation(# param.s w/o given default:
        graph: Graph | DiGraph | MultiDiGraph, k: int, pred_mode: bool, skip_embeddings: bool, # for evaluation & scheduling (see readme)
        device: str, n_epochs_cv: int, n_epochs_pred: int,  # for running model
        n_MLP_layers: int, dim_MLP: int, dim_n2v: int, n_GNN_layers: int, dim_between: int, dim_U: int, n_U_layers: int,  # for constructing model
        p: float, m: int, m_ns: int, batch_size: int, n_batches: int,  # for p-trees & node2vec
        l: int = 8, n_pass: int = 1, scatter_type: str = 'sum', lr_gnn: float = 0.001, lr_n2v: float = 0.01,  # param.s w/ given default
        cutoff: int = 1000  # fixed by exercise
        ) -> None:
    '''runs model in gnn.py thru [k]-fold cross-validation (CV) (pred_mode=False) on given [graph] (designed for CITE) by splitting graph into learning & prediction subgraph (from first [cutoff]=1000 nodes) and then further splitting learning subgraph into shuffled training & validation subgraphs during CV; mean & std of k accuracies from CV get printed; finally/optionally (pred_mode=True) model is trained on learning subgraph and then used to predict node_labels of prediction subgraph; node2vec & closed walk embedding are produced beforehand on the full [graph]'''
    print(f"Device: {device}")
    G_full = Sparse_Graph(graph, False)  # sparse rep. of full graph
    n_nodes = G_full.n_nodes
    nodes_learn = np.arange(cutoff, n_nodes)  # nodes (indices) of learning subgraph
    #n_nodes_learn = n_nodes - cutoff

    # retrieve last & current embedding param.s:
    params_n2v_last, params_cw_last = [], []
    params_n2v_current, params_cw_current = [p, m, m_ns, dim_n2v, batch_size, n_batches, lr_n2v], [l]
    path_n2v, path_cw = 'src/cite/node2vec_parameters.pkl', 'src/cite/closed_walks_parameters.pkl'
    path_X, path_W = 'src/cite/node2vec_embedding.pkl', 'src/cite/closed_walks_embedding.pkl'
    if exists(path_n2v):
        with open(path_n2v, 'rb') as file:
            params_n2v_last = pickle.load(file)
    if exists(path_cw):
        with open(path_cw, 'rb') as file:
            params_cw_last = pickle.load(file)

    ## node2vec:
    if skip_embeddings and (params_n2v_last == params_n2v_current) and exists(path_X):  # skip node2vec if last & current param.s coincide
        with open(path_X, 'rb') as file:
            X = pickle.load(file)
    else:
        print("Train node2vec embedding on random p-trees")
        X = train_node2vec(RT_Iterable(G_full,  # compute node2vec-embedding of full graph
                                       p, m, m_ns, batch_size),
                                       dim_n2v, m, n_batches, batch_size, device, lr_n2v)
        with open(path_n2v, 'wb') as file:  # save/re-write list of N2V-param.s to pkl-file
            pickle.dump(params_n2v_current, file)
        with open(path_X, 'wb') as file:  # save/re-write N2V-embedding tensor to pkl-file
            pickle.dump(X, file)
    X = standard_norm(X)  # standard-normalize node2vec-embedding

    ## closed walks:
    if skip_embeddings and (params_cw_last == params_cw_current) and exists(path_W):  # skip closed walks if last & current param.s coincide
        with open(path_W, 'rb') as file:
            W = pickle.load(file)
    else:
        print("Compute closed walk kernel embedding")
        W = CW_kernel(G_full, l)  # compute CW-embedding of full graph
        with open(path_cw, 'wb') as file:  # save/re-write list of CW-param.s to pkl-file
            pickle.dump(params_cw_current, file)
        with open(path_W, 'wb') as file:  # save/re-write CW-embedding tensor to pkl-file
            pickle.dump(W, file)
    #W = geom_norm(W)  # custom-normalize CW-embedding
    W = log_norm(W)  # compress scale variation of (absolute) CW-embedding using base-2-logarithm

    ## rest of model:
    if pred_mode:  # node_label prediction mode:
        print("Train model & predict unknown node labels\n")
        G_learn = Sparse_Graph(graph.subgraph(nodes_learn), True)  # sparse rep. of learning subgraph
        G_pred = Sparse_Graph(graph.subgraph(np.arange(cutoff)), False)  # sparse rep. of prediction subgraph

        node_labels_pred = run_model(  # node_labels predicted on [cutoff] first nodes (prediction subgraph)
            G_learn, G_pred,  # sparse rep. of learning & prediction subgraph resp.
            X[G_learn.node_idx], X[G_pred.node_idx],  # node2vec embedding slices for learning & prediction resp., split via cutoff
            W[G_learn.node_idx], W[G_pred.node_idx],  # CW embedding slices for learning & prediction resp., split via cutoff
            device, n_epochs_pred, n_MLP_layers, dim_MLP, dim_n2v, n_GNN_layers, dim_between, dim_U, n_U_layers,
            l, n_pass, scatter_type, lr_gnn)

        with open("CITE-Predictions.pkl", "wb") as file:  # save/re-write list of predicted node_labels to pkl-file
            pickle.dump(node_labels_pred, file)
        print(f"\n-> Node label predictions were saved in {file.name}")

    else:  # cross-validation mode:
        print(f"Run {k}-fold cross-validation")
        # for CV: construct node slices for [k] train/val subgraphs/splits of train subgraph & X:
        rng = default_rng(seed=None)
        n_nodes_val = (n_nodes - cutoff) // k  # number of nodes in (regular) validation split/subgraph
        #n_nodes_val = n_nodes // k
        nodes_rest = nodes_learn  # rest nodes: nodes left over from nodes_learn after splitting (an instance of) nodes_val off of it
        splits = []  # split list: collect node split tuples

        for split in range(k - 1):
            #n_nodes_rest -= n_nodes_val
            nodes_val = rng.choice(nodes_rest, size=n_nodes_val, replace=False, p=None, axis=0, shuffle=False)  # sample val split from rest nodes
            nodes_train = np.setdiff1d(nodes_learn, nodes_val, assume_unique=True)  # train split = complement of val split w.r.t. learning subgraph
            nodes_rest = np.setdiff1d(nodes_rest, nodes_val, assume_unique=True)  # subtract val split from rest nodes
            splits.append((nodes_train, nodes_val))  # add train/val split (tuple) to split list
        # add last (k-th) split (w/ len(nodes_learn) >= n_nodes_val) to split list:
        splits.append((np.setdiff1d(nodes_learn, nodes_rest, assume_unique=True), nodes_rest))

        # run [k]-fold CV on model w/ split list:
        accuracies, best_epochs = [], []  # collect accuracies on train vs. val data, and epochs w/ best accuracy in each split
        for split, (nodes_train, nodes_val) in enumerate(splits, start=1):
            print(f"\n\nSplit {split}")
            G_train = Sparse_Graph(graph.subgraph(nodes_train), True)
            G_val = Sparse_Graph(graph.subgraph(nodes_val), True)

            accuracy, best_epoch = run_model(
                G_train, G_val,  # sparse rep. of training & validation subgraph resp.
                X[G_train.node_idx], X[G_val.node_idx],  # node2vec embedding slices for training & validation resp.
                W[G_train.node_idx], W[G_val.node_idx],  # CW embedding slices for training & validation resp.
                device, n_epochs_cv, n_MLP_layers, dim_MLP, dim_n2v, n_GNN_layers, dim_between, dim_U, n_U_layers,
                l, n_pass, scatter_type, lr_gnn)
            accuracies.append(accuracy)
            best_epochs.append(best_epoch)
            print(f"\nAccuracy on split {split} (in %):\t{accuracy * 100}\tBest epoch of split {split}:\t{int(best_epoch + 1)}")

        acc_mean, acc_std = np.mean(accuracies, axis=0), np.std(accuracies, axis=0)
        best_epoch_mean, best_epoch_std = np.mean(best_epochs, axis=0), np.std(best_epochs, axis=0)
        print(f"\n\n-> Mean \u00b1 StD over Accuracies from {k}-fold CV (rounded in %):\t{acc_mean * 100:.2f} \u00b1 {acc_std * 100:.2f}")
        print(f"Sweetspot for number of epochs:\t{best_epoch_mean + 1:.1f} \u00b1 {best_epoch_std:.1f}")

    print(f"\nParameters:")
    print(f"k = {k}, n_epochs_cv = {n_epochs_cv}, n_epochs_pred = {n_epochs_pred}, n_MLP_layers = {n_MLP_layers}, dim_MLP = {dim_MLP}, n_GNN_layers = {n_GNN_layers}, dim_between = {dim_between}, n_U_layers = {n_U_layers}, dim_U = {dim_U},")  # GNN
    print(f"p = {p}, m = {m}, m_ns = {m_ns}, dim_n2v = {dim_n2v}, batch_size = {batch_size}, n_batches = {n_batches},")  # node2vec
    print(f"n_pass = {n_pass}, scatter_type = {scatter_type}, lr_gnn = {lr_gnn}, lr_n2v = {lr_n2v}, l = {l}")  # rest



#print_progress = False
skip_embeddings = False
if __name__ == "__main__":
    # run model evaluation for CITE:
    import argparse
    import pickle
    from timeit import default_timer
    #import torch as th
    from torch.cuda import is_available as cuda_is_available
    from torch.backends.mps import is_available as mps_is_available
    #print_progress = True
    t_start = default_timer()

    # configure parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-cv', '--cross_val', action='store_false',
                        help="run cross-validation mode")  # optional bool argument
    parser.add_argument('-k', '--k', nargs='?', default=12, const=12, type=int,
                        help="number of splits/folds in cross-validation")  # optional argument
    parser.add_argument('-epochs', '--n_epochs', nargs='?', default=25, const=50, type=int,
                        help="number of epochs (per split)")  # optional argument
    parser.add_argument('-s', '--skip', action='store_true',
                        help="skip computation for those embeddings whose parameters have not been modified since the last run(s)")  # optional bool argument

    args = parser.parse_args()  # parse from command line or pass string manually w/ .split()

    # parser-parameters:
    pred_mode = args.cross_val #default: True  # True: train on known part of graph & produce prediction on unknown part, False: run CV on known part & print accuracies
    k = args.k #default: 12
    n_epochs_cv = args.n_epochs #25 #20 #30 #40 #50
    n_epochs_pred = args.n_epochs #25 #20 #30
    # whether or not to skip embedding computations if their resp. last & current param.s coincide (to save trivial compute):
    skip_embeddings = args.skip #default: False

    # other parameters:
    #device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    device = ("cuda" if cuda_is_available() else "mps" if mps_is_available() else "cpu")
    #device = 'cpu'
    n_MLP_layers = 5
    dim_MLP = 200 #30 #50 #100 #200
    dim_n2v = 180 #128 #180 #200 #256 #500
    n_GNN_layers = 5
    dim_between = 200 #15 #30 #50 #65 #100 #200
    dim_U = 200 #15 #30 #50 #65 #100 #200
    n_U_layers = 2
    p = 0.5 #0.1 #0.2 #0.5 #0.8 #1.0  # for p-trees
    m = 10 #5 #7 #10 #20 # for p-trees
    m_ns = 19 #5 #7 #9 #10 #19 #20 # for p-trees
    batch_size = 100 #10 #100 #1000  # for node2vec
    n_batches = 10 #10 #100  # for node2vec

    # param.s w/ default in function argument:
    l = 21 #8 #11 #21  # for CW
    n_pass = 1 #1 #2 #3 #5 #10
    scatter_type = 'sum' #'mean' #'max'
    lr_gnn = 0.001 #0.01 #0.001
    lr_n2v = 0.01 #0.01 #0.001

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    print("---")
    run_evaluation(# param.s w/o given default:
        graph, k, pred_mode, skip_embeddings,  # for evaluation & scheduling (see readme)
        device, n_epochs_cv, n_epochs_pred,  # for running model
        n_MLP_layers, dim_MLP, dim_n2v, n_GNN_layers, dim_between, dim_U, n_U_layers,  # for constructing model
        p, m, m_ns, batch_size, n_batches,  # for p-trees & node2vec
        l, n_pass, scatter_type, lr_gnn, lr_n2v,  # param.s w/ given default
        )
    print(f"Time = {(default_timer() - t_start) / 60} mins\n---")
