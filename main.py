'''The entry point of our implementation'''

# external imports
import argparse
import networkx as nx
import numpy as np
import pickle
import os

from sklearn.model_selection import KFold

import torch as th
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch.optim import Adam#, RMSprop
from torch.utils.data import DataLoader#, TensorDataset
# from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports
from src.dataset import Custom_Dataset
from src.collation import custom_collate
from src.layer import GNN_Layer
from src.pooling import Sum_Pooling
from src.virtual_node import Virtual_Node

import torch.optim as optim
from typing import List, Dict, Any, Tuple
import yaml

from src.model import GNN, EarlyStopping
from src.hpo import hpo as hpt


def main(scatter: list[str], hpo:bool=False) -> None:
    # TODO finish evaluation...
    '''for each parsed [scatter_type]: create model & optimizer object, send everything necessary to [device], train on ZINC_Train & evaluate on every (parsed) [datasets] in several epochs, return mean absolute error (MAE) on every (parsed) dataset'''
    #will use all available GPUs
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    print(f"---\nDevice: {device}\n")  # which device is being used for torch operations

    device = th.device(device)  # set device for torch operations
    best_config = {
        "batch_size":61,
        "beta1":0.9152,
        "beta2":0.999,
        "dim":61,
        "dim_mlp":252,
        "dropout_prob":0.4846,
        "lr":0.002261,
        "lrsched":"cosine",
        "n_gnn_layers":4,
        "n_layers":2,
        "n_mlp_layers":2,
        "n_virtual_layers":2,
        "nonlin":"celu",
        "scatter_type":"sum",
        "trial_number":91,
        "use_dropout":0,
        "use_residual":0,
        "use_skip":1,
        "use_virtual_nodes":0,
        "use_weight_decay":0,
        "weight_decay":0.0000049,
    }



    
    def get_test_data(data:List[nx.Graph]):
        #test data does not have graph labels
        return [graph for graph in data if graph.graph["label"]==None]

    def get_train_data(data:List[nx.Graph]):
        return [graph for graph in data if graph.graph["label"]!=None]

    with open("datasets/HOLU/data.pkl", "rb") as data:
        graphs:List[nx.Graph] = pickle.load(data)
    max_number_nodes:int =max([graph.number_of_nodes() for graph in graphs])
    test_graphs = get_test_data(graphs)
    graphs = get_train_data(graphs)

    if hpo:
        best_config = hpt(graphs, max_n_nodes=max_number_nodes, device=device)
        best_config = best_config | {"lr_sched": "cosine"}
    
    import wandb
    ### Run Model
    cv_mae = [{
        "train":[],
        "val":[],
        "test":0.0
    } for i in range(5)]
    
    with open("src/feature_config.yaml", "r") as f:
        feature_config = yaml.safe_load(f)
    n_circles = feature_config["circle"]["length"]
    n_samples = feature_config["hosoya"]["num_samples"]
    print("Got a maximum of", max_number_nodes, "nodes")
    # construct GNN model of given [scatter_type]
    model = GNN(best_config["scatter_type"], 
        best_config["use_virtual_nodes"], 
        best_config["n_mlp_layers"], 
        best_config["dim_mlp"], 
        best_config["n_virtual_layers"], 
        best_config["n_gnn_layers"], 
        best_config["dim"],
        best_config["dim"], 
        best_config["dim"], 
        best_config["n_layers"], 
        best_config["n_layers"],
        dim_node=35 + 4 + 7 + 6 + max_number_nodes + n_circles + n_samples -3,#based on maximum node count and config: 35 + 4 + 7 + 6 + Circle_length + (num_samples+1) + max_node_count, idk why -3
        dim_edge=5,
        mlp_nonlin=best_config["nonlin"],
        m_nonlin=best_config["nonlin"],
        u_nonlin=best_config["nonlin"],
        skip=best_config["use_skip"],
        residual=best_config["use_residual"],
        dropbout_prob=best_config.get("dropout_prob", 0.0) if best_config["use_dropout"] else 0.0
    )

    splitter = KFold(5)
    test_loader = DataLoader(Custom_Dataset(test_graphs, is_test=True, node_features_size=max_number_nodes, device=device), batch_size=1, shuffle=False, collate_fn=custom_collate) #shuffling makes no sense

    for i, (train_graph_idx, val_graph_idx) in enumerate(splitter.split(list(range(len(graphs))))):
        earlystopper = EarlyStopping(patience=20, verbose=True, delta=10**(-4), mode="min") #use 20 patience but only start using it at epoch 150
        
        train_graphs = [graphs[idx] for idx in train_graph_idx]
        val_graphs = [graphs[idx] for idx in val_graph_idx]

        train_loader = DataLoader(Custom_Dataset(train_graphs, seed=i, node_features_size=max_number_nodes, device=device), batch_size=best_config["batch_size"], shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(Custom_Dataset(val_graphs, node_features_size=max_number_nodes, device=device), batch_size=len(val_graphs), shuffle=True, collate_fn=custom_collate)

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(device)  # move model to device
        
        wandb.init(project="gnn_holu", config= best_config, reinit=True)

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=best_config["lr"], betas=(best_config["beta1"], best_config["beta2"]), weight_decay=best_config["weight_decay"] if best_config["use_weight_decay"] else 0.0)  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...

        if best_config["lrsched"] == "cosine":
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        else:
            scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=best_config["lr"], max_lr=best_config["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=200)

        agg_train_loss:List[float] = []
        val_loss:float = 0.0
        # run training & evaluation phase for [n_epochs]
        for epoch in range(200):
            agg_train_loss = []
            val_loss = 0.0
            model.train()
            # training phase
            # run thru sparse representation of each training batch graph
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in train_loader:
                # set gradients to zero
                optimizer.zero_grad()

                # # move training batch representation to device, should be done already beforehand to make the training faster
                # edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

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
            cv_mae[i]["train"].append(train_loss)

            val_loss:float = 0.0
            model.eval()  # switch model to evaluation mode
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in val_loader:#outputs just one batch with all validation graphs
                with th.no_grad():
                    # move evaluation batch representation to device
                    edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                    # evaluate forward fct. to predict graph labels
                    y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                    val_loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()

            cv_mae[i]["val"].append(val_loss)

            scheduler.step()

            wandb.log({"train_loss": train_loss, "valid_loss": val_loss, "epoch": epoch})

            if epoch >= 150 and earlystopper(val_loss, model).early_stop:
                earlystopper.load_checkpoint(model)
                
                break

        #now do 20 epochs on the combined dataset
        combined_graphs = train_graphs + val_graphs
        combined_loader = DataLoader(Custom_Dataset(combined_graphs, seed=i, node_features_size=max_number_nodes, device=device), batch_size=best_config["batch_size"], shuffle=True, collate_fn=custom_collate)
        for g in optimizer.param_groups:
            g["lr"] = best_config["lr"]/10
        model.train()
        #new scheduler and lr but not a new optimizer
        sched = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        train_loss_agg= []
        for epoch in range(20):
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in combined_loader:
                # set gradients to zero
                optimizer.zero_grad()
                # forward pass and loss
                y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                #print("y_pred:", y_pred.size)
                #print("graph_labels_col:", graph_labels_col.size)
                train_loss = F.l1_loss(y_pred, graph_labels_col, reduction="mean")
                
                # backward pass and sgd step
                train_loss.backward()
                optimizer.step()
                train_loss_agg.append(train_loss.item())

            wandb.log({"train_loss": np.mean(train_loss_agg)})
            sched.step()

        # evaluation phase
        model.eval()  # switch model to evaluation mode
        y_test_pred = []
        # run thru sparse representation of each evaluation batch graph
        for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in test_loader:
            with th.no_grad():
                # move evaluation batch representation to device
                edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                # evaluate forward fct. to predict graph labels
                y_test_pred.append(model(node_features_col, edge_features_col, edge_idx_col, batch_idx))

        wandb.log({"train_loss": np.mean(agg_train_loss), "valid_loss": val_loss})
        wandb.run.summary["final_score"] = val_loss
        wandb.run.summary["state"]="finished"
        wandb.finish(quiet=True)
        cv_mae[i]["test"] = y_test_pred
        
    # Print summary of all final MAEs
    print(f"\n\n---\n\n-> Mean Absolute Errors (rounded) for the HOLU datasets and scatter operation types:\n\nScatter \u2193", end="")
    for i in range(5):
        print(cv_mae[i])


    print("\n\nParameter Values Used:") 
    for key in best_config:
        print(f" - {key} ({type(best_config[key])}): {best_config[key]}")

    #save the best test_pred to file, based on validation score
    for i, cv in enumerate(cv_mae):
        if cv["val"] == min([cv["val"] for cv in cv_mae]):
            with open(f"cv_test_pred.pkl", "wb") as f:
                pickle.dump(cv["test"], f)
            break

if __name__ == "__main__":
    # configure parser
    parser = argparse.ArgumentParser(usage="%(prog)s [options]", description="Run GNN model on ZINC datasets for graph tasks.")  # create parser object

    parser.add_argument('-s', '--scatter', nargs='*', default=['sum', 'mean', 'max'],
                        help="list of predefined [scatter] operation types to be used for message passing in GNN model, called by their resp. names ['sum', 'mean', 'max'] (w/o quotes or brackets, separated by spaces only). If left empty, defaults to calling all of them once in the above order. Names not included will be skipped.")  # optional argument
    
    parser.add_argument("-hpo", "--hpt", action=argparse.BooleanOptionalAction, help="Run hyperparameter optimization using BOHB.", default=False)

    args = parser.parse_args()  # parse from command line  #'-d '.split()  #'-s '.split()
    main(args.scatter, args.hpt)  # run w/ parsed arguments
