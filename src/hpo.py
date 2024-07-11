'''The entry point of our implementation'''
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pickle
from sympy import Number, use
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from .dataset import Custom_Dataset
from .collation import custom_collate
from .model import GNN, EarlyStopping
import json, dataclasses
from pathlib import Path

import os
import wandb
#for wandb logging
os.environ["WANDB_START_METHOD"] = "thread"



from .layer import activation_function
import networkx as nx
import random
import time
def hpo(graphs: list[nx.Graph], max_n_nodes:int, n_trials: int = 100, device:str="cpu"):
    '''hyperparameter optimization via optuna for node classification (Ex.3)'''
    import optuna
    import wandb
    device = th.device(device)

    dataset = Custom_Dataset(graphs, node_features_size=max_n_nodes, device=device)
    data_size = len(graphs)

    def objective(trial:optuna.Trial):
        # define search space for hyperparameters
        #layer & dimensions
        n_gnn_layers = trial.suggest_int("n_gnn_layers", 1, 5)
        dim = trial.suggest_int("dim", 32,512, log=True)
        n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 3)
        dim_mlp = trial.suggest_int("dim_mlp", 64, 512, log=True)
        n_layers = trial.suggest_int("n_layers", 1,3)

        #nodes, training regularization
        use_virtual_nodes = trial.suggest_categorical("use_virtual_nodes", [0,1])
        n_virtual_layers = trial.suggest_int("n_virtual_layers", 1, 3)
        use_skip = trial.suggest_categorical("use_skip", [0,1])
        use_residual = trial.suggest_categorical("use_residual", [0,1])
        droupbout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
        use_dropout = trial.suggest_categorical("use_dropout", [1, 0])
        
        #nonlinearities
        nonlin = trial.suggest_categorical("nonlin", list(activation_function.keys()))

        #other training related
        batch_size = trial.suggest_int("batch_size", 10, 1000, log=True)
        lr = trial.suggest_float("lr", 0.00001, 0.01, log=True)
        lrsched = trial.suggest_categorical("lrsched", ["cosine", "cyclic"])
        beta1 = trial.suggest_float("beta1", 0.9, 0.95)
        beta2 = trial.suggest_categorical("beta2", [0.999])
        weight_decay = trial.suggest_float("weight_decay", 1e-06, 1e-03, log=True)
        use_weight_decay = trial.suggest_categorical("use_weight_decay", [1, 0])

        #scatter operation
        scatter_type = trial.suggest_categorical("scatter_type", ['sum', 'mean', 'max'])

        config = dict(trial.params)
        config["trial_number"] = trial.number
        wandb.init(project="labcourse_5_holu_hpo", config=config)
        if not config["use_weight_decay"]:
            config["weight_decay"] = 0.0
        if not config["use_dropout"]:
            config["dropout_prob"] = 0.0
        import yaml
        with open("src/feature_config.yaml", "r") as f:
            feature_config = yaml.safe_load(f)
        n_circles = feature_config["circle"]["length"]
        n_samples = feature_config["hosoya"]["num_samples"]
        print("Got a maximum of", max_n_nodes, "nodes")
        # construct GNN model of given [scatter_type]
        model = GNN(scatter_type, 
            config["use_virtual_nodes"], 
            config["n_mlp_layers"], 
            config["dim_mlp"], 
            config["n_virtual_layers"], 
            config["n_gnn_layers"], 
            config["dim"],
            config["dim"], 
            config["dim"], 
            config["n_layers"], 
            config["n_layers"],
            dim_node=35 + 4 + 7 + 6 + max_n_nodes + n_circles + n_samples -3,#based on maximum node count and config: 35 + 4 + 7 + 6 + Circle_length + (num_samples+1) + max_node_count, idk why -3
            dim_edge=5,
            mlp_nonlin=config["nonlin"],
            m_nonlin=config["nonlin"],
            u_nonlin=config["nonlin"],
            skip=config["use_skip"],
            residual=config["use_residual"],
            dropbout_prob=config.get("dropout_prob", 0.0)
        )
        
        random.seed(time.time())
        indices = list(range(data_size))
        random.shuffle(indices)
        train_idx,val_idx = indices[:data_size//5], indices[data_size//5:]

        train_loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=custom_collate, sampler=th.utils.data.SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=data_size//5, collate_fn=custom_collate, sampler=th.utils.data.SubsetRandomSampler(val_idx))

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(device)  # move model to device

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"])  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...

        # if config["lrsched"] == "cosine":
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        # else:
        #     scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["lr"], max_lr=config["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=100)

        agg_train_loss:List[float] = []
        val_loss:float = 0.0
        # run training & evaluation phase for [n_epochs]
        for epoch in range(100):
            agg_train_loss = []
            val_loss = 0.0
            # training phase
            # run thru sparse representation of each training batch graph
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in train_loader:
                # set gradients to zero
                optimizer.zero_grad()

                # move training batch representation to device
                edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                # forward pass and loss
                y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                #print("y_pred:", y_pred.size)
                #print("graph_labels_col:", graph_labels_col.size)
                train_loss = F.l1_loss(y_pred, graph_labels_col, reduction="mean")  # graph_labels_col = target vector (y_true)

                # backward pass and sgd step
                train_loss.backward()
                optimizer.step()

                agg_train_loss.append(train_loss.item())
            
            train_loss = np.mean(agg_train_loss)

            val_loss:float = 0.0
            model.eval()  # switch model to evaluation mode
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in val_loader:#outputs just one batch with all validation graphs
                with th.no_grad():
                    # move evaluation batch representation to device
                    edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                    # evaluate forward fct. to predict graph labels
                    y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                    val_loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()

            scheduler.step()
            trial.report(val_loss, epoch)

            if trial.should_prune():
                wandb.log({"train_loss": np.mean(agg_train_loss), "valid_loss": val_loss})
                wandb.run.summary["final_score"] = val_loss
                wandb.run.summary["state"]="pruned"
                wandb.finish(quiet=True)
                raise optuna.TrialPruned()

            wandb.log({"train_loss": train_loss, "valid_loss": val_loss, "epoch": epoch})
                
        wandb.log({"train_loss": np.mean(agg_train_loss), "valid_loss": val_loss})
        wandb.run.summary["final_score"] = val_loss
        wandb.run.summary["state"]="finished"
        wandb.finish(quiet=True)

        return val_loss

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=100, reduction_factor=3
    ))#, pruner=optuna.pruners.MedianPruner()) #<- pruner not used in optimized search
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params
