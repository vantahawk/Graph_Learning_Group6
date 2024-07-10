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

from typing import List, Dict, Any, Tuple
import yaml

from src.model import GNN


def main(scatter: list[str], hpo:bool=False) -> None:
    # TODO finish evaluation...
    '''for each parsed [scatter_type]: create model & optimizer object, send everything necessary to [device], train on ZINC_Train & evaluate on every (parsed) [datasets] in several epochs, return mean absolute error (MAE) on every (parsed) dataset'''
    #will use all available GPUs
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    print(f"---\nDevice: {device}\n")  # which device is being used for torch operations

    device = th.device(device)  # set device for torch operations
    
    if hpo:
        print("Using Hyperparameter Optimization using Smac. This may take a while.")
        from src.hpo import hpt
        param_default = { #these may not be in the configspace, thus we need to add them manually, if they are not they are not used, but still need be specified
            "weight_decay": 1e-06,
            "use_virtual_nodes": 1,
            "n_virtual_layers": 1,
            "dim_U": 30,
        }
        param_default |= hpt(device)
        scatter_type_list = [param_default["scatter_type"]]
    else:
        ### Parameters
        scatter_type_list = []
        for scatter_type in scatter:
            if scatter_type in ['sum', 'mean', 'max']:  # check if parsed [scatter_type] name is included in default list (valid)
                if scatter_type not in scatter_type_list:
                    scatter_type_list.append(scatter_type)
            else:
                pass
        
        param_default:Dict[str, Any] = {
            "batch_size": 32,
            "beta1": 0.9028,
            "beta2": 0.999,
            "dim_M": 128,
            "dim_MLP": 128,
            "dim_U": 128,
            "dim_between": 64,
            "dropout_prob": 0.462,
            "lr":0.001,
            "lrsched": "cosine",
            "m_nlin": "relu",
            "mlp_nlin": "relu",
            "n_GNN_layers": 5,
            "n_M_layers": 1,
            "n_MLP_layers": 3,
            "n_U_layers": 3,
            "n_epochs": 200,
            "n_virtual_layers": 1,
            "scatter_type": "sum",
            "u_nlin": "relu",
            "use_dropout": 1,
            "use_residual": 1,
            "use_skip": 0,
            "use_virtual_nodes": 1,
            "use_weight_decay": 1,
            "weight_decay": 0.0000238
        }
        if len(scatter_type) == 0:
            print("Using mean as scatter operation.")
                
            scatter_type_list = [param_default["scatter_type"]]
    
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
    model = GNN(scatter_type, 
        param_default["use_virtual_nodes"], 
        param_default["n_MLP_layers"], 
        param_default["dim_MLP"], 
        param_default["n_virtual_layers"], 
        param_default["n_GNN_layers"], 
        param_default["dim_between"],
        param_default["dim_M"], 
        param_default["dim_U"], 
        param_default["n_M_layers"], 
        param_default["n_U_layers"],
        dim_node=35 + 4 + 7 + 6 + max_number_nodes + n_circles + n_samples -3,#based on maximum node count and config: 35 + 4 + 7 + 6 + Circle_length + (num_samples+1) + max_node_count, idk why -3
        dim_edge=5,
        mlp_nonlin=param_default["mlp_nlin"],
        m_nonlin=param_default["m_nlin"],
        u_nonlin=param_default["u_nlin"],
        skip=param_default["use_skip"],
        residual=param_default["use_residual"],
        dropbout_prob=param_default.get("dropout_prob", 0.0)
    )

    
    splitter = KFold(5)
    test_loader = DataLoader(Custom_Dataset(test_graphs, is_test=True, node_features_size=max_number_nodes), batch_size=len(test_graphs), shuffle=False, collate_fn=custom_collate) #shuffling makes no sense

    for i, (train_graph_idx, val_graph_idx) in enumerate(splitter.split(graphs)):

        train_graphs = [graphs[idx] for idx in train_graph_idx]
        val_graphs = [graphs[idx] for idx in val_graph_idx]

        train_loader = DataLoader(Custom_Dataset(train_graphs, seed=i, node_features_size=max_number_nodes), batch_size=param_default["batch_size"], shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(Custom_Dataset(val_graphs, node_features_size=max_number_nodes), batch_size=len(val_graphs), shuffle=True, collate_fn=custom_collate)

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(device)  # move model to device
        
        wandb.init(project="gnn_holu", config= param_default | {"scatter_type": scatter_type}, reinit=True)

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=param_default["lr"], betas=(param_default["beta1"], param_default["beta2"]), weight_decay=param_default["weight_decay"] if param_default["use_weight_decay"] else 0.0)  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...

        if param_default["lrsched"] == "cosine":
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=param_default["n_epochs"], eta_min=1e-6)
        else:
            scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=param_default["lr"], max_lr=param_default["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=param_default["n_epochs"])

        agg_train_loss:List[float] = []
        val_loss:float = 0.0
        # run training & evaluation phase for [n_epochs]
        for epoch in range(param_default["n_epochs"]):
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

        # evaluation phase
        model.eval()  # switch model to evaluation mode
        y_pred = None
        # run thru sparse representation of each evaluation batch graph
        for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in test_loader:
            with th.no_grad():
                # move evaluation batch representation to device
                edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                # evaluate forward fct. to predict graph labels
                y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                

        wandb.log({"train_loss": np.mean(agg_train_loss), "valid_loss": val_loss})
        wandb.run.summary["final_score"] = val_loss
        wandb.run.summary["state"]="finished"
        wandb.finish(quiet=True)
        cv_mae[i]["test"] = y_pred
        
    # Print summary of all final MAEs
    print(f"\n\n---\n\n-> Mean Absolute Errors (rounded) for the HOLU datasets and scatter operation types:\n\nScatter \u2193", end="")
    for i in range(5):
        print(cv_mae[i])

    # print("\n\nParameter Values Used:")  # Recap Parameters
    # print(f"train batch size: {batch_size}, # of epochs: {n_epochs}")  # training
    # print(f"# of GNN layers: {n_GNN_layers}, dim. between GNN layers: {dim_between}")  # GNN
    # print(f"# of M layers: {n_M_layers}, hidden dim. of M: {dim_M}")  # M
    # print(f"# of U layers: {n_U_layers}", end="")  # U
    # if n_U_layers > 1:
    #     print(f", hidden dim. of U: {dim_U}", end="")
    # print(f"\nuse virtual nodes: {use_virtual_nodes}", end="")  # virtual nodes
    # if use_virtual_nodes:
    #     print(f", # of VN-MLP layers: {n_virtual_layers}", end="")
    # print(f"\n# of MLP layers: {n_MLP_layers}", end="")  # MLP
    # if n_MLP_layers > 1:
    #     print(f", hidden dim. of MLP: {dim_MLP}")

    # print("---")

    print("\n\nParameter Values Used:") 
    for key in param_default:
        print(f" - {key} ({type(param_default[key])}): {param_default[key]}")


if __name__ == "__main__":
    # configure parser
    parser = argparse.ArgumentParser(usage="%(prog)s [options]", description="Run GNN model on ZINC datasets for graph tasks.")  # create parser object

    parser.add_argument('-s', '--scatter', nargs='*', default=['sum', 'mean', 'max'],
                        help="list of predefined [scatter] operation types to be used for message passing in GNN model, called by their resp. names ['sum', 'mean', 'max'] (w/o quotes or brackets, separated by spaces only). If left empty, defaults to calling all of them once in the above order. Names not included will be skipped.")  # optional argument
    
    parser.add_argument("-hpo", "--hpt", action=argparse.BooleanOptionalAction, help="Run hyperparameter optimization using BOHB.", default=False)

    args = parser.parse_args()  # parse from command line  #'-d '.split()  #'-s '.split()
    main(args.scatter, args.hpt)  # run w/ parsed arguments
