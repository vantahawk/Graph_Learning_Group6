"""The entry point of our graph kernel implementation."""
#internal imports
from re import L
from decorators import parseargs
from utils import load_graphs_dataset, make_data, ValidationLossEarlyStopping as EarlyStopping, unique_file
from hpo import opt_hps
from models import GraphLevelGCN, NodeLevelGCN
from preprocessing import normalized_adjacency_matrix
#external imports
import importlib
from typing import *
import psutil
import numpy as np
import os
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from pandas import DataFrame
from networkx import Graph
from timeit import timeit

# for how to parse args, see the docstring for parseargs
class TorchModel:
    def __init__(self, model:Module, Adj:Tensor, X:np.ndarray, y:np.ndarray, device:str, layers:int, batch_size:int=None, Adj_test:Tensor=None, X_test:np.ndarray=None, y_test:np.ndarray=None, dataset:str=None):
        def is_None(x)->bool:
            return isinstance(x, type(None))
        
        self.model:Module = model
        self.dataset:str = dataset
        self.Adj:Tensor = Adj if not is_None(Adj_test) else Adj[:int(0.9*Adj.shape[0])]
        self.X_train:Tensor = Tensor(X) if not is_None(X_test) else Tensor(X[:int(0.9*X.shape[0])]) #else do a train_test split
        self.y_train:Tensor = Tensor(y) if not is_None(y_test) else Tensor(y[:int(0.9*y.shape[0])])#else do a train_test split

        
        self.Adj_test = Adj_test if not is_None(Adj_test) else Adj[int(0.9*Adj.shape[0]):]
        """10% datasplit"""
        self.X_test = Tensor(X_test) if not is_None(X_test) else Tensor(X[int(0.9*X.shape[0]):])
        """10% datasplit"""
        self.y_test = Tensor(y_test) if not is_None(y_test) else Tensor(y[int(0.9*y.shape[0]):])
        """10% datasplit"""

        # print(self.y_train.shape)

        # determine input and output dimensions
        self.input_dim:int = self.X_train.shape[2]
        self.output_dim:int = self.y_train.shape[len(self.y_train.shape)-1]

        self.device:str = device
        self.layers:int = layers
        self.batch_size:int = batch_size

@parseargs(
    device={
        "default":"best",
        "type":str,
        "help":"The devide to run training and inference on. Either \"cpu\", \"mps\",\"cuda\" or \"best\".",
        "flags":["d"]
    },
    default_hps={
        "default":True,
        "type":bool,
        "help": "Whether to use the default Hyperparameters or optimize for them using Optuna.",
        "flags":["def-hps"]
    },
    rec_times={
        "default":True,
        "type":bool,
        "help":"Wether to record training and inference times.",
        "flags":["t"]
    },
    verbose={
        "default":False,
        "type":bool,
        "help":"whether to print a lot more data while running.",
        "flags":["v"]
    },
    debug={
        "default":False,
        "type":bool,
        "help":"whether to do a debug run, uses less graphs",
        "flags":["db"]
    },
    __description="The entry point of our GCN implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
    __help=True
)
def main(device:str, default_hps:bool, rec_times:bool, verbose:bool, debug:bool):
    #just the calling of the implementations should be in this method.
    level = "graph"
    dataset = "HOLU"
    if device not in ["cpu", "mps", "cuda", "best"]:
        raise ValueError("The selected device is not available. Please select one of: [\"cpu\", \"mps\",\"cuda\", \"best\"]")
    else:
        if device == "best":
            device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(device)

    print(f"Training and Evaluating a GraphLevelGCN on dataset {dataset}.")

    graphs:List[Graph] = load_graphs_dataset(os.path.join("datasets", dataset, "data.pkl"))
    if debug:
        print("Doing a debug run, using 10th of the samples.")
        graphs = graphs[:len(graphs)//10]

    features, labels, test_features, test_feature_idx = make_data(graphs, dataset)

    adjacency_tensors:Tensor = normalized_adjacency_matrix(graphs)
    test_adj_tensors:Tensor = torch.tensor([adjacency_tensors[g] for g, _ in enumerate(graphs) if g in test_feature_idx])
    adjacency_tensors:Tensor = torch.tensor([adjacency_tensors[g] for g, _ in enumerate(graphs) if g not in test_feature_idx])

    test_graphs = [graph for g, graph in enumerate(graphs) if g in test_feature_idx]
    graphs = [graph for g, graph in enumerate(graphs) if g not in test_feature_idx]

    print("Data-Loading and Feature-Preparation successful.")

    if default_hps:
        print("Using preevaluated hyperparameters, will go directly to CV.")
        hyperparams = {
            "epochs":500,
            "batch_size":156 if not debug else 156//3,
            "use_bias": 1,
            "use_dropout": 1,
            "dropout_prob": 0.421177208845452,
            "learning_rate": 0.002372330072992,
            "use_early_stop":0,
            "grad_clip":3.583576452266625,
            "weight_decay":0 #not optimized for
        }
        #builds some logic, but actually just a reuse of the smac hp opt logic
        tmodel = TorchModel(GraphLevelGCN, adjacency_tensors, features, labels, device, layers=5)

    else:
        print("Now optimizing the hyperparams with smac, this may take a while.")
        hyperparams, tmodel = opt_hps(GraphLevelGCN, adjacency_tensors, features, labels, dataset, device, layers=5, intensifier_type="Hyperband")
        

        os.makedirs(f"out/{dataset}", exist_ok=True)
        hparams_out:str = unique_file(f"out/{dataset}/best_params.csv")
        DataFrame(list(hyperparams.values()), index=list(hyperparams.keys())).to_csv(hparams_out)

        print(f"Optimized the hyperparams, saved into \"{hparams_out}\".\nNow verifying the run and reporting accuracies with 10-fold cross-validation.")
    

    cv:KFold = KFold(10, shuffle=True)
    accuracies = []
    times = {"train":[], "test":[]}
    feature_idx = np.arange(features.shape[0]).reshape((-1, 1))
    label_idx = np.arange(labels.shape[0])
    for train_idx, val_idx in cv.split(feature_idx, label_idx):
        
        Adj_train, X_train, y_train = adjacency_tensors[train_idx], Tensor(features[train_idx]), Tensor(labels[train_idx])
        Adj_test, X_test, y_test = adjacency_tensors[val_idx], Tensor(features[val_idx]), Tensor(labels[val_idx])

        Adj_test = Adj_test.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

            
        # create dataset and loader for mini batches
        train_dataset = TensorDataset(Adj_train, X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

        # construct neural network and move it to device
        model = GraphLevelGCN(
            input_dim=tmodel.input_dim, 
            output_dim=tmodel.output_dim, 
            hidden_dim=64, 
            num_layers=tmodel.layers, 
            use_bias=hyperparams["use_bias"], 
            use_dropout=hyperparams["use_dropout"], 
            dropout_prob=hyperparams.get("dropout_prob", 0),
            nonlin="lrelu"
        )
        # print("Model-type:", type(model)) #debug-print

        model.train()
        model.to(device)

        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

        early_stop = EarlyStopping(patience=hyperparams["es_patience"], min_delta=hyperparams["es_min_delta"]) if hyperparams["use_early_stop"] else None

        train_loss:Tensor = None
        val_loss:Tensor = None
        train_time = timeit()
        for epoch in range(hyperparams["epochs"]):
            #type checker
            adj:Tensor
            x:Tensor
            y_true:Tensor
            batch:int = 0
            for adj, x, y_true in train_loader:#batches
                batch +=1
                # set gradients to zero
                opt.zero_grad()

                # move data to device
                x = x.to(device)
                y_true = y_true.to(device)
                adj = adj.to(device)

                # forward pass and loss
                y_pred:Tensor = model(adj, x)
                if torch.isnan(y_pred).any():
                    print("got NANs in the output!!!")
                    print(y_pred)
                train_loss = F.cross_entropy(y_pred, y_true)

                # backward pass and sgd step
                print(f"Train-Loss at E{epoch}/B{batch} -", train_loss)
                # if train_loss > 10 or torch.isnan(train_loss).any():
                #     # print(y_pred, "\nVS\n", y_true, "\n\n")
                
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams["grad_clip"], error_if_nonfinite=True)
                opt.step()

            #we just use the valid_set here for early stopping
            with torch.no_grad():
                y_test_pred = model(Adj_test, X_test)
                val_loss = F.cross_entropy(y_test_pred, y_test)
            if hyperparams["use_early_stop"] and early_stop(val_loss.item()):
                print("Early stopping at epoch:", epoch)
                break
        train_time = timeit() - train_time
        times["train"].append(train_time)

        model.eval()
        test_time = timeit()
        with torch.no_grad():
            y_pred_logits = model(Adj_test, X_test)
            y_pred_probs = F.softmax(y_pred_logits, dim=1) #apply softmax to get class distributions
            y_pred_labels = torch.argmax(y_pred_probs, dim=1)
            y_test_labels = torch.argmax(y_test, dim=1)
            accuracies.append(
                (y_pred_labels == y_test_labels).float().mean().to('cpu').item()
            )
        test_time = timeit() - test_time
        times["test"].append(test_time)

    accuracies = np.array(accuracies)
    train_times = np.array(times["train"])
    test_times = np.array(times["test"])
    print(f"Accuracies:\t MEAN \t STD\n\t\t\t\t{accuracies.mean():.2f}\t{accuracies.std():.2f}")
    if rec_times:
        print(f"Training took {train_times.mean():.2f} ({train_times.std():.2f}) seconds")
        print(f"Testing took {test_times.mean():.2f} ({test_times.std():.2f}) seconds")
        accs_df = DataFrame(np.concatenate((accuracies, train_times, test_times)), axis=1)
    else:
        accs_df = DataFrame(accuracies)
    csv_out:str = unique_file(os.path.join("out", dataset, "cv_results.csv"))
    accs_df.to_csv(csv_out)
    print("Saved the accuracies to:", csv_out)
    print("Ran with the following hyperparameters:", hyperparams)

    


if __name__ == "__main__":
    main()