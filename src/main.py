"""The entry point of our graph kernel implementation."""
#internal imports
from re import L
from sympy import hyper
from decorators import parseargs
from utils import TorchModel, load_graphs_dataset, make_data, opt_with_smac, ValidationLossEarlyStopping as EarlyStopping, unique_file
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
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from pandas import DataFrame
from networkx import Graph

# for how to parse args, see the docstring for parseargs

@parseargs(
    level={
        "default":"graph",
        "type":str,
        "help": "The level on which to do classification. May be either \'graph\' or \'node\'.",
        "flags":["l"]
    },
    device={
        "default":"cpu",
        "type":str,
        "help":"The devide to run training and inference on. Either \"cpu\" or \"cuda\".",
        "flags":["d"]
    },
    dataset={
        "default":None,
        "type":str,
        "help": "Depends on level: If \"graph\"-level then available datasets are \"ENZYMES\" and \"NCI1\", if \"node\"-level, then available datasets are \"Citeseer\" and \"Cora\".",
        "flags":["data", "ds"]
    },
    default_hps={
        "default":False,
        "type":bool,
        "help": "Whether to use the default Hyperparameters or optimize for them using Smac3.",
        "flags":["def-hps"]
    },
    __description="The entry point of our GCN implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
    __help=True
)
def main(level:Literal["graph", "node"], device:str, dataset:str, default_hps:bool):
    #just the calling of the implementations should be in this method.
    if device not in ["cpu", "cuda"]:
        raise ValueError("The selected device is not available. Please select one of: [\"cpu\", \"cuda\"]")
    else:
        device = torch.device(device)
    
    if level not in ["graph", "node"]:
        raise ValueError(f"The selected classification level is not available. Expected on of: [\"graph\", \"node\"], but got \"{level}\".")

    match level:
        case "graph":
            if not dataset:
                dataset="ENZYMES"
            else:#make sure the data is in the available datasets#
                if not dataset in ["ENZYMES", "NCI1"]:
                    raise ValueError(f"The chosen dataset \"{dataset}\" is neither \"ENZYMES\" nor \"NCI1\". Maybe you wanted to do a node-level classification?")

            print("Training and Evaluating a GraphLevelGCN.")

            graphs:List[Graph] = load_graphs_dataset(os.path.join("datasets", dataset, "data.pkl"))

            features, labels = make_data(graphs, dataset)

            adjacency_tensors:Tensor = normalized_adjacency_matrix(graphs)

            print("Data-Loading successfull.")
            

            if default_hps:
                print("Using preevaluated hyperparameters, will go directly to CV.")
                if dataset == "NCI1":
                    hyperparams = {
                        "epochs":500,
                        "batch_size":7,
                        "use_bias": 0,
                        "use_dropout": 0,
                        "dropout_prob": 0.012,
                        "learning_rate": 0.000028241144018,
                        "use_early_stop":0,
                        "es_patience":10,
                        "es_min_delta":0.005,
                        "grad_clip":2.0
                    }
                else:
                    hyperparams = {
                        "epochs":500,
                        "batch_size":7,
                        "use_bias": 0,
                        "use_dropout": 0,
                        "dropout_prob": 0.012,
                        "learning_rate": 0.000028241144018,
                        "use_early_stop":0,
                        "es_patience":10,
                        "es_min_delta":0.005,
                        "grad_clip":2.0
                    }
                #builds some logic, but actually just a reuse of the smac hp opt logic
                tmodel = TorchModel(GraphLevelGCN, adjacency_tensors, features, labels, device, layers=5)

            else:
                print("Now optimizing the hyperparams with smac, this may take a while.")
                hyperparams, tmodel = opt_with_smac(GraphLevelGCN, adjacency_tensors, features, labels, dataset, device, layers=5, intensifier_type="Hyperband")
                

                os.makedirs(f"out/{dataset}", exist_ok=True)
                hparams_out:str = unique_file(f"out/{dataset}/best_params.csv")
                DataFrame(list(hyperparams.values()), index=list(hyperparams.keys())).to_csv(hparams_out)

                print(f"Optimized the hyperparams, saved into \"{hparams_out}\".\nNow verifying the run and reporting accuracies with 10-fold cross-validation.")
            

            cv:KFold = KFold(10, shuffle=True)
            accuracies = []
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
                print("Model-type:", type(model)) #debug-print

                model.train()
                model.to(device)

                # construct optimizer
                opt = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

                early_stop = EarlyStopping(patience=hyperparams["es_patience"], min_delta=hyperparams["es_min_delta"]) if hyperparams["use_early_stop"] else None

                train_loss:Tensor = None
                val_loss:Tensor = None
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
                        if train_loss > 10 or torch.isnan(train_loss).any():
                            print(y_pred, "\nVS\n", y_true, "\n\n")
                        
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

                model.eval()
                with torch.no_grad():
                    y_pred_logits = model(Adj_test, X_test)
                    y_pred_probs = F.softmax(y_pred_logits, dim=1) #apply softmax to get class distributions
                    y_pred_labels = torch.argmax(y_pred_probs, dim=1)
                    y_test_labels = torch.argmax(y_test, dim=1)
                    accuracies.append(
                        (y_pred_labels == y_test_labels).float().mean().to('cpu').item()
                    )
        
            accuracies = np.array(accuracies)
            print(f"Accuracies:\t MEAN \t STD\n\t\t\t\t{accuracies.mean():.2f}\t{accuracies.std():.2f}")
            accs_df = DataFrame(accuracies)
            csv_out:str = unique_file(os.path.join("out", dataset, "cv_results.csv"))
            accs_df.to_csv(csv_out)
            print("Saved the accuracies to:", csv_out)
            print("Ran with the following hyperparameters:", hyperparams)

        case "node":
            if not dataset:
                dataset="Citeseer"
            else:#make sure the data is in the available datasets#
                if not dataset in ["Citeseer", "Cora"]:
                    raise ValueError(f"The chosen dataset \"{dataset}\" is neither \"Citeseer\" nor \"Cora\". Maybe you wanted to do a graph-level classification?")
                
            print("Training and Evaluating a NodeLevelGCN.")
                
            train_graphs = load_graphs_dataset(os.path.join("datasets", dataset+"_Train", "data.pkl"))
            test_graphs =load_graphs_dataset(os.path.join("datasets", dataset+"_Eval", "data.pkl"))

            X_train, y_train = make_data(train_graphs, dataset)
            X_test, y_test = make_data(test_graphs, dataset)
            
            Adj_train:Tensor = normalized_adjacency_matrix(train_graphs)
            Adj_test:Tensor = normalized_adjacency_matrix(test_graphs)

            print("Data-Loading successfull.")
            if default_hps:
                print("Using preevaluated hyperparameters, will go directly to CV.")
                if dataset == "Citeseer":
                    hyperparams = {#
                        "grad_clip":4.554179406698939,
                        "learning_rate":0.003771495836228,
                        "use_bias":0,
                        "use_dropout":1,
                        "dropout_prob":0.400239272036408,
                        "weight_decay":0.000036919472173,
                        "use_early_stop":1,
                        "es_min_delta":0.032974482830919,
                        "es_patience":3,
                        "epochs":500
                    }
                else: #Cora
                    hyperparams={
                        "grad_clip":3.224129213842141,
                        "learning_rate":0.002189964407297,
                        "use_bias":1,
                        "use_dropout":0,
                        "use_early_stop":1,
                        "weight_decay":0.065421243156799,
                        "es_min_delta":0.000322125383191,
                        "es_patience":6,
                        "epochs":500,
                    }
                #builds some logic, but actually just a reuse of the smac hp opt logic
                tmodel = TorchModel(NodeLevelGCN, Adj_train, X_train, y_train, X_test=X_test, y_test=y_test, Adj_test=Adj_test, device=device, layers=3)
            else:
                print("Now optimizing the hyperparams with smac, this may take a while.")
                hyperparams, tmodel = opt_with_smac(NodeLevelGCN,Adj_train, X_train, y_train, dataset, device, layers=3, intensifier_type="SuccessHalving", X_test=X_test, y_test=y_test, Adj_test=Adj_test)
                print("Optimized the hyperparams and saved them. Now verifying the run with repetitions and reporting accuracy.")
                os.makedirs(f"out/{dataset}", exist_ok=True)
                hparams_out:str = unique_file(f"out/{dataset}/best_params.csv")
                DataFrame(list(hyperparams.values()), index=list(hyperparams.keys())).to_csv(hparams_out)

            X_train, y_train = Tensor(X_train), Tensor(y_train)
            X_test, y_test = Tensor(X_test), Tensor(y_test)

            
                    
            # create dataset and loader for mini batches
            train_dataset = TensorDataset(Adj_train, X_train, y_train)
            #TODO: make batch_size depend on 
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            accuracies = []
            #repeat 10 times
            for rep in range(10):
                # construct neural network and move it to device
                model = NodeLevelGCN(
                    input_dim=tmodel.input_dim, 
                    output_dim=tmodel.output_dim, 
                    hidden_dim=64, 
                    num_layers=tmodel.layers, 
                    use_bias=hyperparams["use_bias"], 
                    use_dropout=hyperparams["use_dropout"], 
                    dropout_prob=hyperparams.get("dropout_prob", 0)
                )

                model.train()
                model.to(device)
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                Adj_test = Adj_test.to(device)

                # construct optimizer
                opt = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

                early_stop = EarlyStopping(patience=hyperparams["es_patience"], min_delta=hyperparams["es_min_delta"]) if hyperparams["use_early_stop"] else None

                train_loss:Tensor = None
                val_loss:Tensor = None
                for epoch in range(hyperparams["epochs"]):
                    
                    for adj, x, y_true in train_loader:#batches
                        # set gradients to zero
                        opt.zero_grad()

                        # move data to device
                        x = x.to(device)
                        y_true = y_true.to(device)
                        adj = adj.to(device)

                        # forward pass and loss
                        y_pred:Tensor = model(adj, x)
                        train_loss = F.cross_entropy(y_pred[0], y_true[0])
                        # backward pass and sgd step
                        train_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams["grad_clip"], error_if_nonfinite=True)
                        opt.step()

                    #we just use the valid_set here for early stopping
                    with torch.no_grad():
                        y_test_pred = model(Adj_test, X_test)
                        #get the first graph of the batch and use its loss, we only have one
                        val_loss = F.cross_entropy(y_test_pred[0], y_test[0])

                    if hyperparams["use_early_stop"] and early_stop(val_loss.item()):
                        break

                        
                model.eval()
                with torch.no_grad():
                    y_pred_logits = model(Adj_test, X_test)
                    y_pred_probs = F.softmax(y_pred_logits[0], dim=1) #apply softmax to get class distributions
                    y_pred_labels = torch.argmax(y_pred_probs, dim=1)
                    y_test_labels = torch.argmax(y_test[0], dim=1)
                    accuracies.append(
                        (y_pred_labels == y_test_labels).float().mean().to('cpu').item()
                    )
        
            accuracies = np.array(accuracies)
            print(f"Accuracies:\t MEAN \t STD\n\t\t\t\t{accuracies.mean():.2f}\t{accuracies.std():.2f}")
            outpath = unique_file(os.path.join("out", dataset, "results.csv"))
            DataFrame(accuracies).to_csv(outpath)
            print("Saved the individual accuracies to:", outpath)
            print("Ran with the following hyperparameters:", hyperparams)


        case _:
            raise ValueError(f"Only the levels [\"graph\", \"node\"] are allowed. You inputted: {level}.")
    


if __name__ == "__main__":
    main()