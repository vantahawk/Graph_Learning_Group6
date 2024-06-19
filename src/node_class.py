# external imports
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score#, roc_auc_score
from sklearn.model_selection import cross_validate, KFold

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

    X = train_node2vec(dataset, dim, l, n_batches, batch_size, device, verbose=True, yield_X=False)  # train node2vec embedding, use as inputs to classifier
    y = dataset.node_labels  # node labels as target values for classifier
    print("Build Node2Vec embedding & labels for node classification..., t=%.2f" % (timer() - t))

    # construct classifier object: logistic regression
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores, default param.s otherwise
        penalty='l2', 
        tol=0.0001, 
        C=1.0, 
        max_iter=10000, 
    ) #multi_class='auto'
    
    # return accuracy scores on X vs. y from k-fold cross-validation using classifier
    split = KFold(n_splits=k, shuffle=True, random_state=None)  # k-fold cross-validation split
    scores = cross_validate(classifier, X, y, scoring='accuracy', cv=split, n_jobs=-1,  # runs on all CPU cores
        groups=None, verbose=0, pre_dispatch='2*n_jobs', return_estimator=False,  # default param.s
        return_train_score=True,  # train scores attribute for comparison, else: False for speed up
        error_score='raise')  # raise fitting error
        #return_indices=False #fit_params=None #params=None
    print("Build Logistic Regression classifier for node classification..., t=%.2f" % (timer() - t))

    return np.mean(scores['test_score']), np.std(scores['test_score'])  # return mean & standard deviation of accuracy scores on all k test splits
    #return np.mean(scores['train_score']), np.std(scores['train_score'])  # results on train splits for comparison


def hpo(graph: nx.Graph, n_trials: int = 100, device:str="cpu"):
    '''hyperparameter optimization via optuna for node classification (Ex.3)'''
    import optuna
    import wandb

    def objective(trial):
        # define search space for hyperparameters
        dim = trial.suggest_int("dim", 8, 256, log=True) #categorical, choose some discrete values
        p = trial.suggest_categorical("p", [0.1, 1]) 
        q = trial.suggest_categorical("q", [0.1, 1])
        l = trial.suggest_int("l", 20, 200, log=True)
        l_ns = trial.suggest_int("l_ns", 20, 2000, log=True)
        batch_size = trial.suggest_int("batch_size", 10, 1000, log=True)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log scale
        delta = trial.suggest_float("delta", 1e-5, 1e-1, log=True)  # log scale

        C = trial.suggest_float("C", 1e-2, 1e2, log=True)  # log scale
        
        config = dict(trial.params)
        config["trial_number"] = trial.number
        wandb.init(project="node_classification", config=config, reinit=True)

        dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels)  # custom iterable dataset of pq-walks
        epoch = 0
        max_epochs = 500
        for X in train_node2vec(dataset, dim, l, max_epochs, batch_size, device, lr, delta, verbose=False, yield_X=True): #n_batches = 1, because not used, bc early stopping
            y = dataset.node_labels  # node labels as target values for classifier

            classifier = LogisticRegression(n_jobs=-1,
                penalty='l2', 
                tol=0.0001, 
                C=C, 
                max_iter=10000, 
            )

            split = KFold(n_splits=10, shuffle=True, random_state=None)  # k-fold cross-validation split
            #do the cv explicitly to prune the optuna run if necessary
            scores = cross_validate(classifier, X, y, scoring='accuracy', cv=split, n_jobs=-1,  # runs on all CPU cores
                groups=None, verbose=0, pre_dispatch='2*n_jobs', return_estimator=False,  # default param.s
                return_train_score=True,  # train scores attribute for comparison, else: False for speed up
                error_score='raise')
            
            mean_score = np.mean(scores['test_score'])
            wandb.log({"mean_score": mean_score})
            trial.report(mean_score, epoch)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish()
                raise optuna.TrialPruned()
            epoch += 1

        wandb.run.summary["final_score"] = mean_score
        wandb.run.summary["state"] = "finished"
        wandb.finish(quiet=True)

        return mean_score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


if __name__ == "__main__":
    # test node classification
    import pickle
    import torch as th

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    total_size = 60000
    batch_size = 500
    n_batches = total_size // batch_size
    dim = 128
    p = 1.0
    q = 4.0
    l = 100
    l_ns = 100
    set_node_labels = True

    with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    # mean, std = node_class(graph, dim, p, q, l, l_ns, n_batches, batch_size, device)
    # #print(mean.dtype, std.dtype)
    # print(f"Mean \u00b1 StD of Accuracy Scores, rounded in %:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")

    # test hyperparameter optimization
    hpo(graph, n_trials=100, device=device)
