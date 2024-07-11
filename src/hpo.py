'''The entry point of our implementation'''
import logging
from typing import List, Dict, Any, Tuple
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from more_itertools import last
import numpy as np
import pickle
from sympy import Number, use
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from .dataset import Custom_Dataset
from .collation import custom_collate
from .model import GNN
from smac import MultiFidelityFacade, Scenario, HyperparameterOptimizationFacade as HPOFacade
from smac.multi_objective import ParEGO
from smac.intensifier import SuccessiveHalving, Hyperband as Hyperband_
from smac.runhistory.runhistory import RunHistory
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.conditions import InCondition, EqualsCondition, GreaterThanCondition
import json, dataclasses
from pathlib import Path

import os
import wandb
#for wandb logging
os.environ["WANDB_START_METHOD"] = "thread"



def get_data(batch_size:int, n_devices:int, budget:int, seed:int)->Tuple[DataLoader, DataLoader]:
    ### Preparation
    # open ZINC_Train as list of nx.Graphs
    with open('datasets/ZINC_Train/data.pkl', 'rb') as data:
        train_graphs = pickle.load(data)
    # preprocess ZINC_Train using our [Custom_Dataset] & then collate it into shuffled batch graphs using our [custom_collate] fct.
    train_loader = DataLoader(Custom_Dataset(train_graphs, size=budget, seed=seed), batch_size=n_devices*batch_size, shuffle=True, collate_fn=custom_collate)

    with open('datasets/ZINC_Val/data.pkl', 'rb') as data:
        valid_graphs = pickle.load(data)
    # preprocess ZINC_Train using our [Custom_Dataset] & then collate it into shuffled batch graphs using our [custom_collate] fct.
    valid_loader = DataLoader(Custom_Dataset(valid_graphs), batch_size=len(valid_graphs), shuffle=True, collate_fn=custom_collate)

    return train_loader, valid_loader


from .layer import activation_function

class optObject:
    def __init__(self, device):
        self.device = th.device("cuda")
        self.model = GNN

    def objective(self, config_:Configuration, budget:int, seed:int)->float:
        budget = int(budget)

        # num_devices = th.cuda.device_count() #needed to increase the batch size for multiple devices
        num_devices = 1
        config:Dict[str, Any] = { #these may not be in the configspace, thus we need to add them manually
            "weight_decay": 1e-06,
            "use_virtual_nodes": 1,
            "n_virtual_layers": 1,
            "dim_U": 3,
        } | dict(config_) 

        #wandb init
        wandb.init(project="gnn_zinc", config=config)


        train_loader, valid_loader = get_data(config_['batch_size'], num_devices, budget=budget, seed=seed)

        # construct GNN model of given [scatter_type]
        model = self.model(config["scatter_type"], 
            config["use_virtual_nodes"], 
            config["n_MLP_layers"], 
            config["dim"], 
            config["n_virtual_layers"], 
            config["n_GNN_layers"], 
            config["dim"],
            config["dim"], 
            config["dim"], 
            config["n_layers"], 
            config["n_layers"],
            mlp_nonlin=config["nonlin"],
            m_nonlin=config["nonlin"],
            u_nonlin=config["nonlin"],
            skip=config["use_skip"],
            residual=config["use_residual"],
            dropbout_prob=config.get("dropout_prob", 0.0) #if not use_dropout then 0.0 <- larger space that does not use dropout.
        )

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(self.device)  # move model to device
        

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"] if config["use_weight_decay"] else 0.0) 

        warm_up_epoch = config["n_epochs"]//10
        schedulerSlow = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_up_epoch)
        if config["lrsched"] == "cosine":
            hpscheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100-warm_up_epoch, eta_min=1e-6)
        else:
            hpscheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["lr"], max_lr=config["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=100)
        plat_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08, verbose=False)
        seq_scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, [schedulerSlow, hpscheduler], [warm_up_epoch], config["n_epochs"])


        valid_loss:float = 0
        # run training & evaluation phase for [n_epochs]
        for epoch in range(config["n_epochs"]):
            # training phase
            # run thru sparse representation of each training batch graph
            agg_train_loss:List[float] = []
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in train_loader:
                # set gradients to zero
                optimizer.zero_grad()
                # cast edge_idx to long
                # move training batch representation to device
                edge_idx_col = edge_idx_col.to(self.device)
                node_features_col = node_features_col.to(self.device)
                edge_features_col = edge_features_col.to(self.device)
                graph_labels_col = graph_labels_col.to(self.device)
                batch_idx = batch_idx.to(self.device)


                # forward pass and loss
                y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                #print("y_pred:", y_pred.size)
                #print("graph_labels_col:", graph_labels_col.size)
                train_loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean')  # graph_labels_col = target vector (y_true)
                # backward pass and sgd step
                train_loss.backward()
                optimizer.step()

                # aggregate training loss
                agg_train_loss.append(train_loss.item())

            # evaluation phase
            model.eval()  # switch model to evaluation mode
            
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in valid_loader:
            
                # move evaluation batch representation to device
                edge_idx_col = edge_idx_col.to(self.device)
                node_features_col = node_features_col.to(self.device)
                edge_features_col = edge_features_col.to(self.device)
                graph_labels_col = graph_labels_col.to(self.device)
                batch_idx = batch_idx.to(self.device)

                # evaluate forward fct. to predict graph labels
                with th.no_grad():
                    y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                    valid_loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()  # l1-loss = mean absolute error (MAE)!

            wandb.log({"valid_loss":float(valid_loss), "train_loss":float(np.mean(agg_train_loss))})

            seq_scheduler.step()
            plat_scheduler.step(valid_loss)

        return valid_loss
import networkx as nx
def hpo(graphs: list[nx.Graph], max_n_nodes, n_trials: int = 100, device:str="cpu"):
    '''hyperparameter optimization via optuna for node classification (Ex.3)'''
    import optuna
    import wandb

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
        wandb.init(project="labcourse_5_holu_hpo", config=config, reinit=True)
        if not config["use_weight_decay"]:
            config["weight_decay"] = 0.0
        if not config["use_dropbout"]:
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
            config["n_mlp_layer"], 
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
        import random
        random.seed(trial.id)
        random.shuffle(graphs)
        train_graphs = graphs[:int(len(graphs)*0.8)]
        val_graphs = graphs[int(len(graphs)*0.8):]

        train_loader = DataLoader(Custom_Dataset(train_graphs, node_features_size=max_n_nodes), batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(Custom_Dataset(val_graphs, node_features_size=max_n_nodes), batch_size=len(val_graphs), shuffle=True, collate_fn=custom_collate)

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(device)  # move model to device
        
        wandb.init(project="gnn_holu", config= config | {"scatter_type": scatter_type}, reinit=True)

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"])  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...

        if config["lrsched"] == "cosine":
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        else:
            scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["lr"], max_lr=config["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=100)

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

            wandb.log({"train_loss": train_loss, "valid_loss": val_loss, "epoch": epoch})
                
        wandb.log({"train_loss": np.mean(agg_train_loss), "valid_loss": val_loss})
        wandb.run.summary["final_score"] = val_loss
        wandb.run.summary["state"]="finished"
        wandb.finish(quiet=True)

        return val_loss

    study = optuna.create_study(direction="maximize")#, pruner=optuna.pruners.MedianPruner()) #<- pruner not used in optimized search
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params





