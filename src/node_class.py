# external imports
from networkx import Graph
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# internal imports
from node2vec import train_node2vec
from random_walks import RW_Iterable



def node_class(graph: Graph, p: float, q: float, l: int, l_ns: int, dim: int,  # main parameters, see sheet
               n_batches: int, batch_size: int, device: str, return_train_score: bool,  # extra parameters
               k: int = 10, lr: float = 0.01, set_node_labels: bool = True) -> list[tuple[str, np.float64, np.float64]]:  # default settings
    '''runs node classification (Ex.3) for given [graph] (here: Citeseer & Cora): trains node2vec embedding X first from given parameters, then uses that as input to learn & classify node labels y by running [k]-fold cross-validation w/ logistic regression, returns mean & std of accuracy scores on test splits'''
    print("Prepare dataset") if print_progress else print(end="")
    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset of pq-walks

    print("Train node2vec embedding") if print_progress else print(end="")
    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, lr)  # train node2vec embedding, use as inputs to classifier
    y = dataset.node_labels  # node labels as target values for classifier

    # construct classifier object: logistic regression
    print("Construct classifier") if print_progress else print(end="")
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, l1_ratio=None)

    # return accuracy scores on X vs. y from [k]-fold cross-validation using [classifier]
    print("Compute results") if print_progress else print(end="")
    scores = cross_validate(classifier, X, y, scoring='accuracy', cv=k, n_jobs=-1,  # runs on all CPU cores
        groups=None, verbose=0, pre_dispatch='2*n_jobs', return_estimator=False,  # default param.s
        return_train_score=return_train_score,  # compute scores on train splits for comparison, else: False for speed up
        error_score='raise')  # raise fitting error

    modes = (['train_score', 'test_score'] if return_train_score else ['test_score'])  # choose strings for data modes
    # return mean & standard deviation of accuracy scores over the [k] eval splits, optionally also over the train splits
    return [(mode, np.mean(scores[mode]), np.std(scores[mode])) for mode in modes]



print_progress = False
if __name__ == "__main__":
    # test node classification
    import pickle
    import torch as th

    print_progress = True
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    batch_size = 10 #100 # 1000 #2000 #5000
    # on batch size together w/ param.s below: >=1000 reaches threshold for Cora!, could not reach threshold for Citeseer w/ <=5000...
    n_batches = 100
    dim = 128
    p = 1.0
    q = 1.0
    l = 5
    l_ns = 5
    return_train_score = True

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    results = node_class(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
    print(f"Mean \u00b1 StD of Accuracy Scores, rounded in %:")
    for mode, mean, std in results:
        print(f"{mode}:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")
