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
import argparse


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

    def objective(trial:optuna.Trial):
        # define search space for hyperparameters

        #GENERAL SEARCH
        dim = trial.suggest_categorical("dim", [128]) 
        p = trial.suggest_categorical("p", [0.1, 1]) 
        q = trial.suggest_categorical("q", [0.1, 1])
        l = trial.suggest_categorical("l", [5])
        l_ns = trial.suggest_categorical("l_ns", [5])
        batch_size = trial.suggest_int("batch_size", 10, 10000, log=True)

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
        wandb.init(project="labcourse_node2vec_Cora_fixed", config=config, reinit=True)

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

def main(dataset:str):
    # test node classification
    import pickle
    import torch as th

    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority

    #default hyperparameters
    if dataset == "Citeseer":
        #pq=1-0.1 <- main default
        config = {
            "sched": "linear",
            "C":48.541,
            "batch_size": 9742,
            "delta": 0.00001324,
            "dim": 128,
            "l": 5,
            "l_ns": 5,
            "lr": 0.0968,
            "n_epochs": 200,
            "p": 1,
            "q": 0.1,
        }
        #pq=1-1
        # config = {
        #     "sched": "linear",
        #     "C": 47.417,
        #     "batch_size": 9586,
        #     "delta": 0.00001599,
        #     "dim": 128,
        #     "l": 5,
        #     "l_ns": 5,
        #     "lr": 0.09754,
        #     "n_epochs": 165,
        #     "p": 1,
        #     "q": 1,
        # }
        # pq=0.1-1 <- was not optimised well by hpo so params are just chosen similarly to above
        # config = {
        #     "sched": "linear",
        #     "C": 48,
        #     "batch_size": 9500,
        #     "delta": 0.00001324,
        #     "dim": 128,
        #     "l": 5,
        #     "l_ns": 5,
        #     "lr": 0.09754,
        #     "n_epochs": 200,
        #     "p": 0.1,
        #     "q": 1,
        # }
    elif dataset == "Cora":
        #pq=1-0.1 <- main default
        config = {
            "sched": "plateau",
            "C": 98.533,
            "batch_size": 8726,
            "delta": 0.005616,
            "dim": 128,
            "l": 5,
            "l_ns": 5,
            "lr": 0.006572,
            "n_epochs": 250,
            "p": 1,
            "q": 0.1,
        }
        #pq=1-1
        # config = {
        #     "sched": "linear",
        #     "C": 13.834,
        #     "batch_size": 4201,
        #     "delta": 0.001215,
        #     "dim": 128,
        #     "l": 5,
        #     "l_ns": 5,
        #     "lr": 0.01152,
        #     "n_epochs": 78,
        #     "p": 1,
        #     "q": 1,
        # }
        #pq=0.1-1
        # config = {
        #     "sched": "linear",
        #     "C": 14.703,
        #     "batch_size": 5714,
        #     "delta": 0.0003326,
        #     "dim": 128,
        #     "l": 5,
        #     "l_ns": 5,
        #     "lr": 0.05343,
        #     "n_epochs": 79,
        #     "p": 0.1,
        #     "q": 1,
        # }

    with open(f'datasets/{dataset}/data.pkl', 'rb') as data:
        graph = pickle.load(data)[0]

    #comment out if hpo should be run
    mean, std = node_class(graph, **config, device=device)
    print(f"Mean \u00b1 StD of Accuracy Scores for dataset {dataset}:\n\trounded in %:\t{round(mean * 100 , 2)} \u00b1 {round(std * 100 , 2)}")

    #to run hyperparameter optimization, comment out the above and uncomment the following 
    # hpo(graph, n_trials=100, device=device) #requires wandb login

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Node Classification')
    parser.add_argument('--dataset', type=str, default="Cora", help='Dataset to use for node classification. Options: \"Citeseer\", \"Cora\"')

    dataset = parser.parse_args().dataset
    if dataset not in ["Citeseer", "Cora"]:
        raise ValueError("Dataset not available. Please choose from \"Citeseer\" or \"Cora\"")

    main(dataset)
