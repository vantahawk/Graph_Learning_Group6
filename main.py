'''The entry point of our implementation'''

# external imports
import argparse
#import networkx as nx
import numpy as np
import pickle
import os
#from sklearn.metrics import mean_absolute_error
#from sympy import Line
from sympy import use
import torch as th
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch.optim import Adam#, RMSprop
from torch.utils.data import DataLoader#, TensorDataset
from torch_scatter import scatter_sum, scatter_mean, scatter_max

# internal imports
from src.dataset import Custom_Dataset
from src.collation import custom_collate
from src.layer import GNN_Layer
from src.pooling import Sum_Pooling
from src.virtual_node import Virtual_Node

from typing import List, Dict, Any, Tuple

from src.model import GNN


def main(datasets: list[str], scatter: list[str], hpo:bool=False) -> None:
    # TODO finish evaluation...
    '''for each parsed [scatter_type]: create model & optimizer object, send everything necessary to [device], train on ZINC_Train & evaluate on every (parsed) [datasets] in several epochs, return mean absolute error (MAE) on every (parsed) dataset'''

    #device = 'cpu'
    #device = 'cuda'
    #will use all available GPUs
    # device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    device="cuda"
    print(f"---\nDevice: {device}\n")  # which device is being used for torch operations

    device = th.device(device)  # set device for torch operations
    
    if hpo:
        print("Using Hyperparameter Optimization using Smac. This may take a while.")
        from src.hpo import hpt
        param_default = { #these may not be in the configspace, thus we need to add them manually
            "weight_decay": 1e-06,
            "use_virtual_nodes": 1,
            "n_virtual_layers": 1,
            "dim_U": 3,
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
        if len(scatter_type) == 0:
            print("Using mean as scatter operation.")
                
            scatter_type_list = ["mean"]
            
        param_default:Dict[str, Any] = {
            "n_GNN_layers": 1,
            "dim_between": 3,
            "dim_M": 3,
            "dim_U": 3, # inactive for n_U_layers <= 1
            "n_M_layers": 1,
            "n_U_layers": 1,
            "use_virtual_nodes": True, # inactive for n_GNN_layers <= 1
            "n_virtual_layers": 1,
            "n_MLP_layers": 1,
            "dim_MLP": 3, # inactive for n_MLP_layers <= 1
            "batch_size": 10, # 10 seems promising, 100 still okay, smaller/larger may take longer
            "n_epochs": 20, 
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "use_weight_decay" : False,
            "scatter_type":"mean"
        }
    
    ### Preparation
    # open ZINC_Train as list of nx.Graphs
    with open('datasets/ZINC_Train/data.pkl', 'rb') as data:
        graphs = pickle.load(data)
    # preprocess ZINC_Train using our [Custom_Dataset] & then collate it into shuffled batch graphs using our [custom_collate] fct.
    train_loader = DataLoader(Custom_Dataset(graphs), batch_size=param_default["batch_size"], shuffle=True, collate_fn=custom_collate)

    with open('datasets/ZINC_Test/data.pkl', 'rb') as data:
        test_graphs = pickle.load(data)
    # preprocess ZINC_Train using our [Custom_Dataset] & then collate it into shuffled batch graphs using our [custom_collate] fct.
    test_loader = DataLoader(Custom_Dataset(test_graphs), batch_size=len(test_graphs), shuffle=True, collate_fn=custom_collate)



    ### Run Model
    MAE_table = []
    for scatter_type in scatter_type_list:  # run thru list of valid scatter-types
        print(f"\n\nScatter-Type: {scatter_type}")

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
            mlp_nonlin=param_default["mlp_nlin"],
            m_nonlin=param_default["m_nlin"],
            u_nonlin=param_default["u_nlin"],
            skip=param_default["use_skip"]
        )

        # if th.cuda.is_available() and th.cuda.device_count() > 1:
        #     model = th.nn.DataParallel(model) # parallelize GNN model for multi-GPU training
        model.train()  # switch model to training mode
        model.to(device)  # move model to device
        

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=param_default["lr"], betas=(param_default["beta1"], param_default["beta2"]), weight_decay=param_default["weight_decay"] if param_default["use_weight_decay"] else 0.0)  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...
        
        warm_up_epoch = param_default["n_epochs"]//10
        schedulerSlow = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_up_epoch)
        if param_default["lrsched"] == "cosine":
            hpscheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100-warm_up_epoch, eta_min=1e-6)
        else:
            hpscheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=param_default["lr"], max_lr=param_default["lr"]*5, step_size_up=2000, mode='triangular2', last_epoch=100)
            
        plat_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08, verbose=False)
        seq_scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, [schedulerSlow, hpscheduler], [warm_up_epoch], param_default["n_epochs"])

        train_loss:th.Tensor = th.tensor(0.0, device=device)  # initialize training loss
        # run training & evaluation phase for [n_epochs]
        for epoch in range(param_default["n_epochs"]):
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

            val_loss:float = 0.0
            model.eval()  # switch model to evaluation mode
            for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in val_loader:
                with th.no_grad():
                    # move evaluation batch representation to device
                    edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                    # evaluate forward fct. to predict graph labels
                    y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                    val_loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()

            plat_scheduler.step(val_loss)
            seq_scheduler.step()

        # evaluation phase
        model.eval()  # switch model to evaluation mode
        MAE = 0.0
        # run thru sparse representation of each evaluation batch graph
        for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in test_loader:
            with th.no_grad():
                # move evaluation batch representation to device
                edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                # evaluate forward fct. to predict graph labels
                y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                MAE = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()

        MAE_table.append((train_loss.item(), val_loss, MAE))

            

    # Print summary of all final MAEs
    print(f"\n\n---\n\n-> Mean Absolute Errors (rounded) for the chosen ZINC datasets and scatter operation types:\n\nScatter \u2193", end="")
    for i in range(len(scatter_type_list)):
        print(f"\n{scatter_type_list[i]}:\t {MAE_table[i]}", end="")

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

    parser.add_argument('-d', '--datasets', nargs='*', default=['Train', 'Val', 'Test'],
                        help="DEPRECATED! list of predefined ZINC_[datasets] to be called by their resp. names ['Train', 'Val', 'Test'] (w/o quotes or brackets, separated by spaces only). Runs evaluation (Ex.6) of each called dataset. If left empty, defaults to calling all of them once in the above order. Names not included will be skipped. Training is always done on ZINC_Train.")  # optional argument

    parser.add_argument('-s', '--scatter', nargs='*', default=['sum', 'mean', 'max'],
                        help="list of predefined [scatter] operation types to be used for message passing in GNN model, called by their resp. names ['sum', 'mean', 'max'] (w/o quotes or brackets, separated by spaces only). If left empty, defaults to calling all of them once in the above order. Names not included will be skipped.")  # optional argument
    
    parser.add_argument("-hpo", "--hpt", action=argparse.BooleanOptionalAction, help="Run hyperparameter optimization using BOHB.", default=False)

    args = parser.parse_args()  # parse from command line  #'-d '.split()  #'-s '.split()
    main(args.datasets, args.scatter, args.hpt)  # run w/ parsed arguments
