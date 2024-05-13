"""The entry point of our graph kernel implementation."""
#internal imports
from decorators import parseargs
from utils import load_graphs_dataset, make_data, opt_with_smac, ValidationLossEarlyStopping as EarlyStopping
from models import GraphLevelGCN, NodeLevelGCN
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
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame

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
    __description="The entry point of our GCN implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
    __help=True
)
def main(level:Literal["graph", "node"], device:str, dataset:str):
    #just the calling of the implementations should be in this method.
    if device not in ["cpu", "cuda"]:
        raise ValueError("The selected device is not available. Please select one of: [\"cpu\", \"cuda\"]")
    
    if level not in ["graph", "node"]:
        raise ValueError(f"The selected classification level is not available. Expected on of: [\"graph\", \"node\"], but got \"{level}\".")

    match level:
        case "graph":
            from .models import GraphLevelGCN as Model
            if not dataset:
                dataset="ENZYMES"
            else:#make sure the data is in the available datasets#
                if not dataset in ["ENZYMES", "NCI1"]:
                    raise ValueError(f"The chosen dataset \"{dataset}\" is neither \"ENZYMES\" nor \"NCI1\". Maybe you wanted to do a node-level classification?")

            print("Training and Evaluating a NodeLevelGCN.")

            graphs = load_graphs_dataset(os.path.join("datasets", dataset, "data.pkl"))

            features, labels = make_data(graphs, dataset)

            print("Data-Loading successfull.")
            print("Now optimizing the hyperparams with smac, this may take a while.")

            hyperparams, tmodel = opt_with_smac(GraphLevelGCN, features, labels, dataset, device, layers=5, intensifier_type="SuccessHalving")

            print("Optimized the hyperparams. Now verifying the run and reporting accuracies with 10-fold cross-validation.")

            cv:StratifiedKFold = StratifiedKFold(10, shuffle=True)
            accuracies = []
            for train_idx, val_idx in cv.split(features, labels):
                X_train, y_train = Tensor(features[train_idx]), Tensor(labels[train_idx])
                X_test, y_test = Tensor(features[val_idx]), Tensor(labels[val_idx])
                    
                # create dataset and loader for mini batches
                train_dataset = TensorDataset(X_train, y_train)
                #TODO: make batch_size depend on 
                train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

                # construct neural network and move it to device
                model = tmodel.model(
                    input_dim=tmodel.input_dim, 
                    output_dim=tmodel.output_dim, 
                    hidden_dim=64, 
                    num_layers=tmodel.layers, 
                    use_bias=hyperparams["use_bias"], 
                    use_dropout=hyperparams["use_dropout"], 
                    dropout_prob=hyperparams["dropout_prob"]
                )

                model.train()
                model.to(tmodel.device)

                # construct optimizer
                opt = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

                early_stop = EarlyStopping(patience=hyperparams["es_patience"], min_delta=hyperparams["es_min_delta"])

                train_loss:Tensor = None
                val_loss:Tensor = None
                for epoch in range(hyperparams["epochs"]):
                    
                    for x, y_true in train_loader:#batches
                        # set gradients to zero
                        opt.zero_grad()

                        # move data to device
                        x = x.to(tmodel.device)
                        y_true = y_true.to(tmodel.device)

                        # forward pass and loss
                        y_pred:Tensor = model(x)
                        train_loss = F.cross_entropy(y_pred, y_true)

                        #we just use the valid_set here for early stopping
                        with torch.no_grad():
                            y_test_pred = model(X_test)
                            val_loss = F.cross_entropy(y_test_pred, y_test)

                        if hyperparams["use_early_stop"] and early_stop(val_loss.item()):
                            break

                        # backward pass and sgd step
                        train_loss.backward()
                        opt.step()

                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test)
                    accuracies.append(
                        (y_pred == y_test).float().mean().item()
                    )
        
            accuracies = np.array(accuracies)
            accs_df = DataFrame(accuracies)
            accs_df.to_csv(os.path.join("out", dataset))
            print("Saved the accuracies to:", os.path.join("out", dataset))
            print("Ran with the following hyperparameters:", hyperparams)

        case "node":
            from .models import NodeLevelGCN as Model
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

            print("Data-Loading successfull.")
            print("Now optimizing the hyperparams with smac, this may take a while.")

            hyperparams, tmodel = opt_with_smac(NodeLevelGCN, X_train, y_train, dataset, device, layers=3, intensifier_type="SuccessHalving", X_test=X_test, y_test=y_test)

            print("Optimized the hyperparams. Now verifying the run and reporting accuracy.")

            X_train, y_train = Tensor(X_train), Tensor(y_train)
            X_test, y_test = Tensor(X_test), Tensor(y_test)
                    
            # create dataset and loader for mini batches
            train_dataset = TensorDataset(X_train, y_train)
            #TODO: make batch_size depend on 
            train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

            # construct neural network and move it to device
            model = tmodel.model(
                input_dim=tmodel.input_dim, 
                output_dim=tmodel.output_dim, 
                hidden_dim=64, 
                num_layers=tmodel.layers, 
                use_bias=hyperparams["use_bias"], 
                use_dropout=hyperparams["use_dropout"], 
                dropout_prob=hyperparams["dropout_prob"]
            )

            model.train()
            model.to(tmodel.device)

            # construct optimizer
            opt = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

            early_stop = EarlyStopping(patience=hyperparams["es_patience"], min_delta=hyperparams["es_min_delta"])

            train_loss:Tensor = None
            val_loss:Tensor = None
            for epoch in range(hyperparams["epochs"]):
                
                for x, y_true in train_loader:#batches
                    # set gradients to zero
                    opt.zero_grad()

                    # move data to device
                    x = x.to(tmodel.device)
                    y_true = y_true.to(tmodel.device)

                    # forward pass and loss
                    y_pred:Tensor = model(x)
                    train_loss = F.cross_entropy(y_pred, y_true)

                    #we just use the valid_set here for early stopping
                    with torch.no_grad():
                        y_test_pred = model(X_test)
                        val_loss = F.cross_entropy(y_test_pred, y_test)

                    if hyperparams["use_early_stop"] and early_stop(val_loss.item()):
                        break

                    # backward pass and sgd step
                    train_loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                accuracies.append(
                    (y_pred == y_test).float().mean().item()
                )
        
            accuracies = np.array(accuracies)
            accs_df = DataFrame(accuracies)
            accs_df.to_csv(os.path.join("out", dataset))
            print("Saved the accuracies to:", os.path.join("out", dataset))
            print("Ran with the following hyperparameters:", hyperparams)


        case _:
            raise ValueError(f"Only the levels [\"graph\", \"node\"] are allowed. You inputted: {level}.")
    


if __name__ == "__main__":
    main()