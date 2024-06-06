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

from typing import *


from smac import MultiFidelityFacade, initial_design, Scenario, HyperparameterOptimizationFacade as HPOFacade
from smac.multi_objective import ParEGO
from smac.intensifier import SuccessiveHalving, Hyperband, AbstractIntensifier
from smac.runhistory.runhistory import RunHistory
from ConfigSpace import Integer, Float, Categorical#,Categorical #<- does not work for Bool rn, use the integer instead
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.conditions import InCondition, EqualsCondition
import warnings
import json


class GNN(Module):
    '''module for the overall GNN, including series of [n_GNN_layers] GNN layers, each w/ optional virtual node, followed by a sum pooling layer & finally a graph-lvl MLP w/ [n_MLP_layers] layers; input dim. is composed of node & edge feature dim., output dim. is 1, matching the scalar, real-valued graph labels'''
    def __init__(self, scatter_type: str, use_virtual_nodes: bool,
                 n_MLP_layers: int, dim_MLP: int, n_virtual_layers: int,
                 n_GNN_layers: int, dim_between: int, dim_M: int, dim_U: int, n_M_layers: int, n_U_layers: int,
                 dim_node: int = 21, dim_edge: int = 3, dim_graph: int = 1) -> None:
        super().__init__()

        self.n_GNN_hidden = n_GNN_layers - 1  # number of hidden GNN layers
        self.n_MLP_hidden = n_MLP_layers - 1  # number of hidden MLP layers
        #self.use_virtual_nodes = use_virtual_nodes  # whether to use (True) or bypass (False) all virtual nodes

        self.GNN_input = GNN_Layer(scatter_type, dim_node, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)  # input GNN layer
        """
        # list of hidden GNN layers, i.e. GNN layers after input GNN layer
        self.GNN_hidden = ModuleList([GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)
                                      for layer in range(self.n_GNN_hidden)])

        if use_virtual_nodes:  # optional list of virtual nodes, one before each hidden GNN layer
            self.virtual_nodes = ModuleList([Virtual_Node(dim_between, n_virtual_layers) for layer in range(self.n_GNN_hidden)])
        """

        # list of hidden GNN layers, i.e. GNN layers after input GNN layer, optionally each w/ prior virtual node
        if use_virtual_nodes:
            GNN_hidden = []
            for layer in range(self.n_GNN_hidden):
                GNN_hidden += [Virtual_Node(dim_between, n_virtual_layers),
                               GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)]
            self.n_GNN_hidden *= 2  # double to account for added virtual nodes in forward fct.
        else:  # disable/bypass virtual nodes
            GNN_hidden = [GNN_Layer(scatter_type, dim_between, dim_edge, dim_M, dim_U, dim_between, n_M_layers, n_U_layers)
                          for layer in range(self.n_GNN_hidden)]
        self.GNN_hidden = ModuleList(GNN_hidden)

        self.sum_pooling = Sum_Pooling()  # sparse sum pooling layer, node- to graph-lvl

        # linear, graph-lvl MLP layers
        if n_M_layers > 1:  # for >=2 MLP layers
            # list of hidden MLP layers including prior input MLP layer
            self.MLP_hidden = ModuleList(
                [Linear(dim_between, dim_MLP, bias=True)] + [Linear(dim_MLP, dim_MLP, bias=True) for layer in range(n_MLP_layers - 2)])
            self.MLP_output = Linear(dim_MLP, dim_graph, bias=True)  # output MLP layer
        else:  # n_MLP_layers <= 1
            self.MLP_hidden = ModuleList([])  # no hidden MLP layers
            self.MLP_output = Linear(dim_between, dim_graph, bias=True)  # singular output MLP layer


    def forward(self, node_features: th.Tensor, edge_features: th.Tensor, edge_idx: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
        '''forward fct. of overall GNN, takes in node & edge features as well as edge index lists & batch_idx of graphs in given batch/dataset, returns predicted graph labels thereof'''
        y = self.GNN_input(node_features, edge_features, edge_idx, batch_idx)  # apply input GNN layer
        """
        if self.use_virtual_nodes:
            for layer in range(self.n_GNN_hidden):
                y = self.virtual_nodes[layer](y, batch_idx)  # apply virtual nodes
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # apply hidden GNN layers
        else:  # disable/bypass virtual nodes
            for layer in range(self.n_GNN_hidden):
                y = self.GNN_hidden[layer](y, edge_features, edge_idx)  # only apply hidden GNN layers
    	"""

        # apply hidden GNN layers w/ optional virtual nodes (version w/o if-statement in forward fct.)
        for layer in range(self.n_GNN_hidden):
            y = self.GNN_hidden[layer](y, edge_features, edge_idx, batch_idx)

        y = self.sum_pooling(y, batch_idx)  # apply sum pooling layer

        for layer in range(self.n_MLP_hidden):
            y = self.MLP_hidden[layer](y)  # apply hidden MLP layers
            y = F.relu(y)  # TODO try different activation fct.s...

        return self.MLP_output(y)  # apply linear output MLP layer



def main(datasets: list[str], scatter: list[str]) -> None:
    # TODO finish evaluation...
    '''for each parsed [scatter_type]: create model & optimizer object, send everything necessary to [device], train on ZINC_Train & evaluate on every (parsed) [datasets] in several epochs, return mean absolute error (MAE) on every (parsed) dataset'''

    #device = 'cpu'
    #device = 'cuda'
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    print(f"---\nDevice: {device}\n")  # which device is being used for torch operations


    ### Parameters
    # GNN
    n_GNN_layers = 1 #1 #2 #3 #5 #10
    dim_between = 3 #5
    dim_M = 3 #5 #6 #12
    dim_U = 3 #5 #10  # inactive for n_U_layers <= 1
    n_M_layers = 1 #2
    n_U_layers = 1 #2
    # virtual node
    use_virtual_nodes = True  # True/False, inactive for n_GNN_layers <= 1
    n_virtual_layers = 1
    # MLP
    n_MLP_layers = 1 #2
    dim_MLP = 3  # inactive for n_MLP_layers <= 1
    # training
    batch_size = 10 #1 #10 #100 #1000 #10000  # 10 seems promising, 100 still okay, smaller/larger may take longer
    n_epochs = 20 #100

    param_default:Dict[str, Any] = {
        "n_GNN_layers": 1,
        "dim_between": 3,
        "dim_M": 3,
        "dim_U": 3,
        "n_M_layers": 1,
        "n_U_layers": 1,
        "use_virtual_nodes": True,
        "n_virtual_layers": 1,
        "n_MLP_layers": 1,
        "dim_MLP": 3,
        "batch_size": 10,
        "n_epochs": 20,
        "lr": 0.001
    }

    T = TypeVar('T')
    param_ranges:Dict[str, Tuple[T, T] | List[Any]] = {
        "n_GNN_layers": [1, 2, 3, 5, 10],
        "dim_between": [3, 5],
        "dim_M": [3, 5, 6, 12],
        "dim_U": [3, 5, 10],
        "n_M_layers": [1, 2],
        "n_U_layers": [1, 2],
        "use_virtual_nodes": [True, False],
        "n_virtual_layers": [1],
        "n_MLP_layers": [1, 2],
        "dim_MLP": [3],
        "batch_size": (1, 10000), #log sampling
        "n_epochs": (20, 20),
        "lr": (0.00001, 0.01) #log sampling
    }

    ### Preparation
    # open ZINC_Train as list of nx.Graphs
    with open('datasets/ZINC_Train/data.pkl', 'rb') as data:
        graphs = pickle.load(data)
    # preprocess ZINC_Train using our [Custom_Dataset] & then collate it into shuffled batch graphs using our [custom_collate] fct.
    train_loader = DataLoader(Custom_Dataset(graphs), batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    datasets_valid = []  # collect parsed dataset names included in default list (valid)
    eval_loader_list = []  # collect preprocessed, valid datasets (DataLoader objects) for evaluation
    datasets_invalid = []  # collect parsed dataset names not included in default list (invalid)
    for dataset in datasets:  # run thru parsed or default list of [datasets]
        if dataset in ['Train', 'Val', 'Test']:  # check if parsed [dataset] name is included in default list (valid)
            if dataset not in datasets_valid:
                datasets_valid.append(dataset)
                # open given [dataset] as list of nx.Graphs
                with open('datasets/ZINC_' + dataset + '/data.pkl', 'rb') as data:
                    graphs = pickle.load(data)
                # preprocess given [dataset] using our [Custom_Dataset], collate it into *singular* batch graph using our [custom_collate] fct. & add it to eval_loader_list
                eval_loader_list.append(DataLoader(Custom_Dataset(graphs), batch_size=len(graphs), shuffle=False, collate_fn=custom_collate))
                #eval_loader_list.append(DataLoader(Custom_Dataset(graphs), batch_size=batch_size, shuffle=False, collate_fn=custom_collate))  # use batch_size instead of singular batch
        else:
            if dataset not in datasets_invalid:
                datasets_invalid.append(dataset)

    if datasets_invalid != []:
        print(f"Invalid dataset names {datasets_invalid} included in command line argument. Will be ignored.")

    scatter_type_list = []
    scatter_type_list_invalid = []
    for scatter_type in scatter:
        if scatter_type in ['sum', 'mean', 'max']:  # check if parsed [scatter_type] name is included in default list (valid)
            if scatter_type not in scatter_type_list:
                scatter_type_list.append(scatter_type)
        else:
            if scatter_type not in scatter_type_list_invalid:
                scatter_type_list_invalid.append(scatter_type)

    if datasets_invalid != []:
        print(f"Invalid scatter-type names {scatter_type_list_invalid} included in command line argument. Will be ignored.")

    if (datasets_valid != []) & (scatter_type_list != []):
        print(f"The ZINC datasets {datasets_valid} will be evaluated using the scatter operation types {scatter_type_list}.")


    ### Run Model
    MAE_table = []
    for scatter_type in scatter_type_list:  # run thru list of valid scatter-types
        print(f"\n\nScatter-Type: {scatter_type}")

        # construct GNN model of given [scatter_type]
        model = GNN(scatter_type, use_virtual_nodes, n_MLP_layers, dim_MLP, n_virtual_layers, n_GNN_layers, dim_between, dim_M, dim_U,n_M_layers, n_U_layers)
        model.to(device)  # move model to device
        model.train()  # switch model to training mode

        # construct optimizer
        optimizer = Adam(model.parameters(), lr=param_default["lr"])  # TODO try diff. optimizers, parameters to be investigated, tested, chosen...

        # run training & evaluation phase for [n_epochs]
        for epoch in range(n_epochs):
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
                loss = F.l1_loss(y_pred, graph_labels_col, reduction='mean')  # graph_labels_col = target vector (y_true)

                # backward pass and sgd step
                loss.backward()
                optimizer.step()

            # evaluation phase
            model.eval()  # switch model to evaluation mode
            MAE_list = []
            print(f"\nepoch {epoch}:\t", end="")

            for i in range(len(datasets_valid)):  # run thru given, valid list of DataLoader objects of [datasets] for evaluation
                # run thru sparse representation of each evaluation batch graph
                for edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx in eval_loader_list[i]:
                    with th.no_grad():
                        # move evaluation batch representation to device
                        edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = edge_idx_col.to(device), node_features_col.to(device), edge_features_col.to(device), graph_labels_col.to(device), batch_idx.to(device)

                        # evaluate forward fct. to predict graph labels
                        y_pred = model(node_features_col, edge_features_col, edge_idx_col, batch_idx)
                        MAE = F.l1_loss(y_pred, graph_labels_col, reduction='mean').item()  # l1-loss = mean absolute error (MAE)!

                    print(f"MAE_{datasets_valid[i]}: {round(MAE, 4)}\t\t", end="")  # print rounded MAE progress for each given dataset

                if epoch == n_epochs - 1:
                    MAE_list.append(MAE)  # after epochs, append final MAE, associated w/ its resp. dataset

        MAE_table.append(MAE_list)

    # Print summary of all final MAEs
    print(f"\n\n---\n\n-> Mean Absolute Errors (rounded) for the chosen ZINC datasets and scatter operation types:\n\nScatter \u2193 | Dataset \u2192\t", end="")
    for dataset in datasets_valid:
        print(f"{dataset}\t", end="")
    for i in range(len(scatter_type_list)):
        print(f"\n{scatter_type_list[i]}:\t", end="")
        for j in range(len(datasets_valid)):
            print(f"{round(MAE_table[i][j], 2)}\t", end="")

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
                        help="list of predefined ZINC_[datasets] to be called by their resp. names ['Train', 'Val', 'Test'] (w/o quotes or brackets, separated by spaces only). Runs evaluation (Ex.6) of each called dataset. If left empty, defaults to calling all of them once in the above order. Names not included will be skipped. Training is always done on ZINC_Train.")  # optional argument

    parser.add_argument('-s', '--scatter', nargs='*', default=['sum', 'mean', 'max'],
                        help="list of predefined [scatter] operation types to be used for message passing in GNN model, called by their resp. names ['sum', 'mean', 'max'] (w/o quotes or brackets, separated by spaces only). If left empty, defaults to calling all of them once in the above order. Names not included will be skipped.")  # optional argument
    
    parser.add_argument("-hpo", "--hyperparameter_optimization", action="store_true", help="Run hyperparameter optimization using BOHB.", default=False)

    args = parser.parse_args()  # parse from command line  #'-d '.split()  #'-s '.split()
    main(args.datasets, args.scatter)  # run w/ parsed arguments
