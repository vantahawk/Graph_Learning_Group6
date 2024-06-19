# external imports
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score#, roc_auc_score
from sklearn.model_selection import cross_validate#, cross_val_score

# internal imports
from node2vec import Node2Vec, train_node2vec
from random_walks import RW_Iterable
from timeit import default_timer as timer


# TODO finish node classification module
def node_class(graph: nx.Graph, dim: int, p: float, q: float, l: int, l_ns: int,  # main parameters, see sheet
               n_batches: int, batch_size: int, device: str,  # extra parameters
               k: int = 10, set_node_labels: bool = True) -> tuple[np.float64, np.float64]:  # default settings
    '''runs node classification (Ex.3) for given [graph] (here: Citeseer & Cora), trains node2vec embedding X first from given parameters, then uses that as input to learn & classify node labels y by running [k]-fold cross-validation w/ logistic regression, returns mean & std of accuracy scores on test splits'''
    t = timer()
    print("Starting node classification..., t=0.0")
    print("Graph has %d nodes and %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))
    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset of pq-walks
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device)  # train node2vec embedding, use as inputs to classifier
    y = dataset.node_labels  # node labels as target values for classifier
    print("Build Node2Vec embedding & labels for node classification..., t=%.2f" % (timer() - t))
    # construct classifier object: logistic regression
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=10000, verbose=0, warm_start=False, l1_ratio=None) #multi_class='auto'
    print("Build Logistic Regression classifier for node classification..., t=%.2f" % (timer() - t))
    # return accuracy scores on X vs. y from k-fold cross-validation using classifier
    scores = cross_validate(classifier, X, y, scoring='accuracy', cv=k, n_jobs=-1,  # runs on all CPU cores
        groups=None, verbose=0, pre_dispatch='2*n_jobs', return_estimator=False,  # default param.s
        return_train_score=True,  # train scores attribute for comparison, else: False for speed up
        error_score='raise')  # raise fitting error
        #return_indices=False #fit_params=None #params=None
    print("Finished node classification. t=%.2f" % (timer() - t))

    return np.mean(scores['test_score']), np.std(scores['test_score'])  # return mean & standard deviation of accuracy scores on all k test splits
    #return np.mean(scores['train_score']), np.std(scores['train_score'])  # results on train splits for comparison



if __name__ == "__main__":
    # test node classification
    import pickle
    import torch as th

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 2000
    n_batches = 100
    dim = 128
    p = 1.0
    q = 2.0
    l = 100
    l_ns = 100
    set_node_labels = True

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    mean, std = node_class(graph, dim, p, q, l, l_ns, n_batches, batch_size, device)
    #print(mean.dtype, std.dtype)
    print(f"Mean \u00b1 StD of Accuracy Scores, rounded in %:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")
