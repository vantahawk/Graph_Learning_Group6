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


def node_class(graph: nx.Graph, dim: int, p: float, q: float, l: int, l_ns: int,  # main parameters, see sheet
               n_epochs: int, batch_size: int, lr:float, sched:str, delta:float,  #learning parameters
               C: float,  # classifier parameter
               device: str,  # extra parameters
               k: int = 10, set_node_labels: bool = True) -> tuple[np.float64, np.float64]:  # default settings
    '''runs node classification (Ex.3) for given [graph] (here: Citeseer & Cora), trains node2vec embedding X first from given parameters, then uses that as input to learn & classify node labels y by running [k]-fold cross-validation w/ logistic regression, returns mean & std of accuracy scores on test splits
    
    Args:
        graph (nx.Graph): input graph
        dim (int): embedding dimension
        p (float): return parameter
        q (float): in-out parameter
        l (int): walk length
        l_ns (int): number of walks
        n_epochs (int): number of epochs
        batch_size (int): batch size
        lr (float): learning rate
        sched (str): learning rate scheduler
        delta (float): early stopping delta
        C (float): logistic regressen classifier parameter
        device (str): device to run on
        k (int): number of folds for cross-validation
        set_node_labels (bool): whether to set node labels for classification

    Returns:
        tuple[np.float64, np.float64]: mean & std of accuracy scores on test splits
    '''
    t = timer()
    print("Starting node classification..., t=0.0")
    print("Graph has %d nodes and %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))
    dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels=True)  # custom iterable dataset of pq-walks

    X = train_node2vec(dataset, dim, l, n_epochs, batch_size, device, lr=lr, delta=delta, lrsched=sched, verbose=True, yield_X=False)  # train node2vec embedding, use as inputs to classifier
    y = dataset.node_labels  # node labels as target values for classifier
    print("Build Node2Vec embedding & labels for node classification..., t=%.2f" % (timer() - t))

    # construct classifier object: logistic regression
    classifier = LogisticRegression(n_jobs=-1,  # runs on all CPU cores
        penalty='l2', 
        tol=0.0001, 
        C=C, 
        max_iter=10000, 
    )
    
    # return accuracy scores on X vs. y from k-fold cross-validation using classifier
    split = KFold(n_splits=k, shuffle=True, random_state=None)  # k-fold cross-validation split
    scores = cross_validate(classifier, X, y, scoring='accuracy', cv=split, n_jobs=-1, error_score='raise')
    print("Build Logistic Regression classifier and did CV for node classification..., t=%.2f" % (timer() - t))

    return np.mean(scores['test_score']), np.std(scores['test_score'])  # return mean & standard deviation of accuracy scores on all k test splits


def hpo(graph: nx.Graph, n_trials: int = 100, device:str="cpu"):
    '''hyperparameter optimization via optuna for node classification (Ex.3)'''
    import optuna
    import wandb

    def objective(trial):
        # define search space for hyperparameters

        #GENERAL SEARCH
        dim = trial.suggest_int("dim", 8, 256, log=True) #categorical, choose some discrete values
        p = trial.suggest_categorical("p", [0.1, 1]) 
        q = trial.suggest_categorical("q", [0.1, 1])
        l = trial.suggest_int("l", 20, 200, log=True)
        l_ns = trial.suggest_int("l_ns", 20, 2000, log=True)
        batch_size = trial.suggest_int("batch_size", 10, 1000, log=True)

        n_epochs = trial.suggest_int("n_epochs", 50, 500, log=True)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log scale
        sched = trial.suggest_categorical("sched", ["constant", "linear", "cosine", "step", "plateau"])  # categorical
        delta = trial.suggest_float("delta", 1e-5, 1e-1, log=True)  # log scale

        C = trial.suggest_float("C", 1e-2, 1e2, log=True)  # log scale

        #OPTIMIZED SEARCH for Citeseer <- was used to squeeze out the last bit of performance
        # dim = trial.suggest_int("dim", 128, 256, log=True) #categorical, choose some discrete values
        # p = trial.suggest_categorical("p", [0.1, 1])
        # q = trial.suggest_categorical("q", [0.1, 1])
        # l = trial.suggest_int("l", 50, 150, log=True)
        # l_ns = trial.suggest_int("l_ns", 10, 100, log=True)
        # batch_size = trial.suggest_int("batch_size", 500, 1000, log=True)

        # n_epochs = trial.suggest_int("n_epochs", 200, 500, log=True)
        # lr = trial.suggest_float("lr", 0.01, 0.1, log=True)  # log scale
        # sched = trial.suggest_categorical("sched", ["constant", "linear", "cosine", "step", "plateau"])  # categorical
        # delta = trial.suggest_float("delta", 0.01, 0.1, log=True)  # log scale

        # C = trial.suggest_float("C", 0.1, 10, log=True)  # log scale
        
        config = dict(trial.params)
        config["trial_number"] = trial.number
        wandb.init(project="node_classification_Cora", config=config, reinit=True)

        dataset = RW_Iterable(graph, p, q, l, l_ns, batch_size, set_node_labels=True)  # custom iterable dataset of pq-walks
        epoch = 0
        max_epochs = n_epochs
        for X in train_node2vec(dataset, dim, l, max_epochs, batch_size, device, lr=lr, delta=delta, lrsched=sched, verbose=False, yield_X=True): #n_batches = 1, because not used, bc early stopping
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
            #not used in optimized search
            # if trial.should_prune():
            #     wandb.run.summary["state"] = "pruned"
            #     wandb.run.summary["final_score"] = mean_score
            #     wandb.finish()
            #     raise optuna.TrialPruned()
            epoch += 1

        wandb.run.summary["final_score"] = mean_score
        wandb.run.summary["state"] = "finished"
        wandb.finish(quiet=True)

        return mean_score

    study = optuna.create_study(direction="maximize")#, pruner=optuna.pruners.MedianPruner()) #<- pruner not used in optimized search
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


if __name__ == "__main__":
    # test node classification
    import pickle
    import torch as th

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority

    dataset = "Citeseer"

    if dataset == "Citeseer":
        config = {
            "sched": "cosine",
            "C":2.884,
            "batch_size": 660,
            "delta": 0.01423,
            "dim": 251,
            "l": 99,
            "l_ns": 22,
            "lr": 0.01907,
            "n_epochs": 500,
            "p": 0.1,
            "q": 1,
        }
    elif dataset == "Cora":
        config = {
            "sched": "constant",
            "C": 0.6516312844703022,
            "batch_size": 257,
            "delta": 0.0785058599417405,
            "dim": 237,
            "l": 71,
            "l_ns": 138,
            "lr": 0.009707123531453407,
            "n_epochs": 461,
            "p": 0.1,
            "q": 1,
        }

    with open(f'datasets/{dataset}/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    #comment out if hpo should be run
    mean, std = node_class(graph, **config, device=device)
    print(f"Mean \u00b1 StD of Accuracy Scores for dataset {dataset}:\n\trounded in %:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")

    # run hyperparameter optimization
    #requires wandb login
    # hpo(graph, n_trials=100, device=device)
