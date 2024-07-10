from pathlib import Path
from networkx import Graph
import networkx as nx
import pickle
from typing import List,Union, Tuple, Any, Callable
import os
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import itertools

import ctypes as ct
from preprocessing import cwk_node_contributions
import yaml
import random 

# from smac import MultiFidelityFacade, Scenario
# from smac.intensifier.hyperband import Hyperband
# from smac.intensifier.successive_halving import SuccessiveHalving
# from smac.facade import AbstractFacade
# from ConfigSpace import ConfigurationSpace, Configuration, Integer, Float, Categorical
# from ConfigSpace.hyperparameters import IntegerHyperparameter
# from ConfigSpace.conditions import InCondition, LessThanCondition

from joblib import Parallel, delayed

Shape = Union[Tuple[int], Union[int, int], Union[int, int, int]]

def decorate_shape(shape)->Callable:
    def wrapper(func:Callable[[Any], np.ndarray])->np.ndarray:
        def wrapped(*args, **kwargs):
            res = func(*args, **kwargs)
            if len(res.shape) != len(shape):
                raise ValueError(f"Expected shape with {len(shape)}, but got {res.shape}")
            if any([res.shape[i] > shape[i] for i in range(len(shape))]):
                raise ValueError(f"Expected shape {shape} or smaller, but got {res.shape}")
            
            #pad zeros to every dimension that is smaller
            for i in range(len(shape)):
                if shape[i] > res.shape[i]:
                    res = np.pad(res, [(0, shape[i]-res.shape[i])], mode="constant")
            return res
        return wrapped
    return wrapper

def load_graphs_dataset(dataset_path:str|Path)->List[Graph]:
    """Loads a pickled dataset and returns it. The expected dataset is a list of networkx.Graph instances.
    
    Args:
    - dataset_path (str|Path): The path to the pkled object. If a Path object is parsed, only allows for posix-paths

    Returns:
    List[Graph] - The unpickled list of Graphs and the Labels (may be a vector or int) 
    
    The labels are set as attributes to the graph objects, access via: `graph.graph["label"]`.
    The node-labels are set as data on the nodes. Access via `graph.nodes(data='node_label')[node_id]`, other data may be accessed alike.
    """
    
    if not dataset_path.endswith(".pkl"):
        dataset_path = os.path.join(dataset_path, "data.pkl")

    def file_not_exists(path):
        print(f"Specified dataset-path \"{dataset_path}\" does not exist.")
        raise FileNotFoundError(f"Specified dataset-path \"{dataset_path}\" does not exist.")
    
    if isinstance(dataset_path, Path):
        if not dataset_path.exists():
            file_not_exists(dataset_path.absolute())
        
        dataset_path = dataset_path.as_posix()
    else:
        if not os.path.exists(dataset_path):
            file_not_exists(dataset_path)

    graphs:List[Graph]
    with open(dataset_path, "rb") as data:
        graphs = pickle.load(data)

    return graphs


def calculate_hosoya_index(graph:Graph, node:int, dist:int, size:int, seed:int=0)->int:
    """Calculates the hosoya index for a graph subset, with bfs starting at node, up to a certain distance and size."""
    def bfs(graph:Graph, start:int, dist:int, size:int, seed:int)->List[int]:
        random.seed(seed)

        visited = [False]*graph.number_of_nodes()
        queue = [start]
        visited[start] = True
        level = 0
        while level < dist and sum(visited) < size:
            new_queue = []
            for q in queue:
                neighbors = list(graph.neighbors(q))
                random.shuffle(neighbors)
                for n in neighbors:
                    if not visited[n]:
                        visited[n] = True
                        new_queue.append(n)
            queue = new_queue
            level += 1
            
        return visited

    def hosoya(graph:Graph, nodes_bool:List[int])->int:
        """Calculates the hosoya index for a graph subset."""
        #get the subgraph
        nodes = np.arange(graph.number_of_nodes())[nodes_bool]
        subgraph = graph.subgraph(nodes)
        return nx.maximal_matching(subgraph)

    sub = bfs(graph, node, dist, size, seed)
    return len(hosoya(graph, sub))


def make_data(graphs:List[Graph], dataset:str, feature_config_path:str="src/feature_config.yaml")->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a list of graphs into a feature matrix and a label vector.

    Args:
    - graphs (List[Graph]): The list of graphs to convert.
    - dataset (str): The dataset to convert. Currently only supports "HOLU".

    Returns:
    Tuple[np.ndarray, np.ndarray]: The feature matrix and the label vector.
    
    FOR HOLU DATASET:
        - The node labels are one-hot encoded. -> 35
        - The 3D vectors for each atom are converted to zmatrix. -> 7
        - The helpful features for each atom are loaded: electronegativities, ionization energies, electron affinities. -> 6
        - Circle finding is done up to 12 atoms. -> 12
        - Hosoya index calculations are done for fixed-size node-centered random subsets of the graph. -> 5
        - The edge features are incorporated as node features. -> 4*max_node_count (4 labels for each possible edge)

    => features.shape == [num_graphs, max_node_count, 65+4*max_node_count];
    => labels.shape == [num_graphs, 1] #the homo/lumo bandgap energy

    HYPERPARAMETERS FOR THE FEATURES:
    - circle: length[Int], cut_duplicates[Bool]
    - hosoya: num_samples[Int], size[Int], depth[Int] (size and depth are similar in effect: O(size^2)~=O(depth))

    """
    features, labels = None, None
    test_features, labels_ = None, None

    match dataset:

        case "HOLU":
            #prep 
            from geomConvert import zmat_from_molecule
            with open(feature_config_path, "r") as f:
                feature_config = yaml.safe_load(f)
            print("Using feature config:", feature_config)
            max_node_count = max([graph.number_of_nodes() for graph in graphs])
            print("Max node count is:", max_node_count)
            
            """ENCODE THE NODE LABELS: ATOM TYPES"""
            #we have the node labels, that must be one-hot encoded: [1..35]
            node_labels = np.zeros((len(graphs), max_node_count, 35))
            for g in range(node_labels.shape[0]):
                for n, nname in enumerate(graphs[g].nodes(data="node_label")):
                    node_labels[g, n, nname[1]-1] = 1

            """MAKE ZMATRIX PER GRAPH"""
            #we have 3d vectors for each atom, that must be converted to zmatrix
            #question: how to handle the missing values in the zmatrix at the beginning? set to 0 to not perturb the normalization
            zmatrices = np.zeros((len(graphs), max_node_count, 7))
            
            #fill in the shape if dome molecules are smaller
            # make_zmatrix = decorate_shape((max_node_count, 7))(zmat_from_molecule)
            # zmatrices = np.array(Parallel(n_jobs=-1)(delayed(make_zmatrix)(graph) for graph in graphs))
            for g in range(zmatrices.shape[0]):
                zmatrices[g][:graphs[g].number_of_nodes()] = zmat_from_molecule(graphs[g])

            """MAKE HELPFUL ATOM FEATURES"""
            #further load the helpful features for each atom: electro-negativities and the first 4 ionization energies, and 0th energy(affinity)
            with open("helpful_extra_features/ionizationenergies.yaml", "r") as f:
                ionization_energies = yaml.load(f, Loader=yaml.FullLoader)

            with open("helpful_extra_features/electronegativities.yaml", "r") as f:
                electronegativities = yaml.load(f, Loader=yaml.FullLoader)
            
            with open("helpful_extra_features/electronaffinities.yaml", "r") as f:
                electronaffinities = yaml.load(f, Loader=yaml.FullLoader)

            help_features = np.zeros((len(graphs), max_node_count, 6))
            for g in range(help_features.shape[0]):
                for n, nname in enumerate(graphs[g].nodes(data="node_label")):
                    atom_type = nname[1]
                    help_features[g, n] = np.array([
                        electronegativities[atom_type], 
                        electronaffinities[atom_type], 
                        *ionization_energies[atom_type]
                    ])

            """BASICALLY A CLOSED WALK KERNEL BUT WITH NOT CONTRIBUTIONS: CIRCLES EACH NODE CONTRIBUTES TO"""
            #further do circle finding, see ex. 1: important for benzene rings and similar structures.
            #do this up to 12 atoms, longer circles are not that important in small molecules
            circles = np.zeros((
                len(graphs), 
                max_node_count, 
                feature_config["circle"]["length"])
            )
            circles = np.array(Parallel(n_jobs=-1)(
                delayed(cwk_node_contributions)(
                    graph=graph, 
                    max_length=feature_config["circle"]["length"], 
                    array_size=max_node_count, 
                    cut_duplicates=feature_config["circle"]["cut_duplicates"]
                ) for graph in graphs
            ))

            """BASICALLY A HOSOYA INDEX KERNEL ON RANDOM NODE CENTERED SUBGRAPHS"""
            #maybe do hosoya index calculations for fixed-size subsets of the graph
            #e.g. do a bfs from each node up to a certain length/size found, then calculate hosoya index, and attribute it to the node
            #use min_distance=2, min_size=5, whenever one is reached stop
            hosoya_indexes = np.zeros((
                len(graphs), 
                max_node_count,
                feature_config["hosoya"]["num_samples"]+1
            ))
            for g in range(hosoya_indexes.shape[0]):
                for n in range(graphs[g].number_of_nodes()):
                    for sample in range(feature_config["hosoya"]["num_samples"]):
                        hosoya_indexes[g, n, sample] = calculate_hosoya_index(
                            graph=graphs[g], 
                            node=n, 
                            dist=feature_config["hosoya"]["depth"], 
                            size=feature_config["hosoya"]["size"],
                            seed=sample
                        )
            #add the whole graph hosoya index as well
            for g in range(hosoya_indexes.shape[0]):
                for n in range(hosoya_indexes.shape[1]):
                    hosoya_indexes[g, n, -1] = len(nx.maximal_matching(graphs[g]))

            """BASICALLY EDGE FEATURES BUT AS NODE FEATURES"""
            #we must incorporate the given edge features somehow as node features
            #is just [0..3] for all edges, means the bond type
            #for each node, add the full edge feature vector of the node
            edge_features = np.zeros((len(graphs), max_node_count, max_node_count*4))
            for g in range(edge_features.shape[0]):
                for n, nname in enumerate(graphs[g].nodes):
                    for edge in graphs[g].edges(nname, data=True):
                        edge_features[g, n, list(graphs[g].nodes).index(edge[1])+edge[2]["edge_label"]] = 1


            #of course normalize everything to [0,1], that is not already normalized (e.g. the one-hot vectors)
            #meaning the zmatrices, the help_features, the circles, the hosoya_indexes
            zmatrices = (zmatrices - zmatrices.min()) 
            zmatrices = zmatrices / zmatrices.max()

            help_features = (help_features - help_features.min())
            help_features = help_features / help_features.max()

            circles = (circles - circles.min())
            circles = circles / circles.max()

            hosoya_indexes = (hosoya_indexes - hosoya_indexes.min())
            hosoya_indexes = hosoya_indexes / hosoya_indexes.max()

            #concatenate all features
            features = np.concatenate([node_labels, zmatrices, help_features, circles, hosoya_indexes, edge_features], axis=2)
            print("Got Feature with shape:", features.shape)

            labels_ = np.array([graph.graph["label"] if graph.graph["label"] is not None else np.nan for graph in graphs])
        case _:
            raise ValueError(f"The given dataset {dataset} is not known, cannot be converted into data.")

    #features where labels are none or nan
    test_features = features[np.isnan(labels_)]
    features = features[~np.isnan(labels_)]
    labels = labels_[~np.isnan(labels_)].astype(np.int64)
    test_feature_idx = np.where(np.isnan(labels_)!=0)[0]

    return features, labels, test_features, test_feature_idx

#somewhere from stack overflow <- basically the keras implementation of early stopping
class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def __call__(self, validation_loss)->bool:
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False

# class TorchModel:
#     @property
#     def configspace(self) -> ConfigurationSpace:

#         cs = ConfigurationSpace(seed=0)

#         # epochs = Integer("epochs", (50, 500), default=100)

#         batch_size = Integer("batch_size", (1, 10000), default=500, log=True)

#         use_bias = Integer("use_bias", (0, 1), default=0) #bool

#         use_dropout = Integer("use_dropout", (0, 1), default=0) #boool
#         dropout_prob = Float("dropout_prob", (0.01, 0.8), default=0.2)

#         learning_rate = Float("learning_rate", (0.00001, 0.01), default=0.005, log=True)

#         use_early_stop = Integer("use_early_stop", (0, 1), default=1) #bool
#         es_patience = Integer("es_patience", (1, 10), default=5)
#         es_min_delta = Float("es_min_delta", (0.0001, 0.1), default = 0.005, log=True)

#         grad_clip = Float("grad_clip", (1.0, 5.0), default=2.0)
#         weight_decay = Float("weight_decay", (10**(-6), 10**(0)), log=True)

#         #if use_dropout is set, use the dropout_prob hp
#         #this saves search space
#         use_drop_prob = InCondition(dropout_prob, use_dropout, [True])
#         # use_epochs = LessThanCondition(epochs, epochs, 0)#change if we use this as a hyperparameter and not use a budget for this
#         use_es_pat = InCondition(es_patience, use_early_stop, [True])
#         use_es_md = InCondition(es_min_delta, use_early_stop, [True])

#         cs.add(use_bias, use_dropout, dropout_prob, learning_rate, use_early_stop, es_patience, es_min_delta, grad_clip, weight_decay)
#         if not self.batch_size:
#             cs.add(batch_size)
#         cs.add(use_drop_prob, use_es_pat, use_es_md)

#         return cs

#     def __init__(self, model:Module, Adj:Tensor, X:np.ndarray, y:np.ndarray, device:str, layers:int, batch_size:int=None, Adj_test:Tensor=None, X_test:np.ndarray=None, y_test:np.ndarray=None, dataset:str=None):
#         def is_None(x)->bool:
#             return isinstance(x, type(None))
        
#         self.model:Module = model
#         self.dataset:str = dataset
#         self.Adj:Tensor = Adj if not is_None(Adj_test) else Adj[:int(0.9*Adj.shape[0])]
#         self.X_train:Tensor = Tensor(X) if not is_None(X_test) else Tensor(X[:int(0.9*X.shape[0])]) #else do a train_test split
#         self.y_train:Tensor = Tensor(y) if not is_None(y_test) else Tensor(y[:int(0.9*y.shape[0])])#else do a train_test split

        
#         self.Adj_test = Adj_test if not is_None(Adj_test) else Adj[int(0.9*Adj.shape[0]):]
#         """10% datasplit"""
#         self.X_test = Tensor(X_test) if not is_None(X_test) else Tensor(X[int(0.9*X.shape[0]):])
#         """10% datasplit"""
#         self.y_test = Tensor(y_test) if not is_None(y_test) else Tensor(y[int(0.9*y.shape[0]):])
#         """10% datasplit"""

#         # print(self.y_train.shape)

#         # determine input and output dimensions
#         self.input_dim:int = self.X_train.shape[2]
#         self.output_dim:int = self.y_train.shape[len(self.y_train.shape)-1]

#         self.device:str = device
#         self.layers:int = layers
#         self.batch_size:int = batch_size


#     def train(self, config:Configuration, seed:int=0, budget:int=50):
#         budget = int(budget)
#         seed = int(seed)

#         config_dict = dict(config)

#         # create dataset and loader for mini batches
#         train_dataset = TensorDataset(self.Adj, self.X_train, self.y_train )
#         #TODO: make batch_size depend on 
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size if self.batch_size!=None else config_dict["batch_size"], shuffle=True)

#         # construct neural network and move it to device
#         model:Module = self.model(
#             input_dim=self.input_dim, 
#             output_dim=self.output_dim, 
#             hidden_dim=64, 
#             num_layers=self.layers, 
#             use_bias=config_dict["use_bias"], 
#             use_dropout=config_dict["use_dropout"], 
#             dropout_prob=config_dict.get("dropout_prob", 0.0)
#         )

#         model.train()
#         model.to(self.device)

#         # construct optimizer
#         opt = torch.optim.Adam(model.parameters(), lr=config_dict["learning_rate"], weight_decay=config_dict["weight_decay"])


#         early_stop = ValidationLossEarlyStopping(patience=config_dict["es_patience"], min_delta=config_dict["es_min_delta"]) if config_dict["use_early_stop"] else None

#         self.Adj_test = self.Adj_test.to(self.device)
#         self.X_test = self.X_test.to(self.device)
#         self.y_test = self.y_test.to(self.device)


#         train_loss:Tensor = None
#         val_loss:Tensor = None
#         for epoch in range(budget):
            
#             for adj, x, y_true in train_loader:#batches
#                 # set gradients to zero
#                 opt.zero_grad()

#                 # move data to device
#                 x = x.to(self.device)
#                 y_true = y_true.to(self.device)
#                 adj = adj.to(self.device)

#                 # forward pass and loss
#                 y_pred:Tensor = model(adj, x)
                
#                 if self.dataset in ["Cora", "Citeseer"]:
#                     train_loss = F.cross_entropy(y_pred[0], y_true[0])
#                 else:
#                     train_loss = F.cross_entropy(y_pred, y_true)

#                 # if train_loss.item() > 1.0:
#                 #     #if our loss is weird, throw it away, somehow the -log must have gotten small values?
#                 #     continue
                
#                 # backward pass and sgd step
#                 train_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config_dict["grad_clip"], error_if_nonfinite=True)
#                 opt.step()

#              #we just use the valid_set here for early stopping
#             with torch.no_grad():
#                 y_test_pred:Tensor = model(self.Adj_test, self.X_test)
#                 if self.dataset in ["Cora", "Citeseer"]:
#                     val_loss = F.cross_entropy(y_test_pred[0], self.y_test[0])
#                 else:
#                     val_loss = F.cross_entropy(y_test_pred, self.y_test)

#                 # acc= (y_test_pred == self.y_test).float().mean().item()
#                 # print("Current Accuracy:", acc)

#             if config_dict["use_early_stop"] and early_stop(val_loss.item()):
#                 break

#         #use the final validation loss as decision factor when comparing configurations
#         print("Solved an smac opt step, got val_loss:", val_loss.item())
#         return val_loss.item()

# def opt_with_smac(model:Module, Adj:Tensor, X:np.ndarray, y:np.ndarray, dataset:str, device:str, layers:int, intensifier_type:Literal["SuccessHalving", "Hyperband"]="SuccessiveHalving", Adj_test:Tensor=None, X_test:np.ndarray=None, y_test:np.ndarray=None)->Tuple[Dict[str, Any], TorchModel]:
#     """#TODO Write this docstring
    
#     Args:
#         - model (Module) : a pytorch module object, not an instance!
#         - Adj (Tensor): the precomputed adjacency tensor of all graphs in the data
#         - X (np.ndarray): the features, if X_test the train features
#         - y (np.ndarray): the labels, if y_test the train labels#
#         - dataset (str): the dataset used
#         - device (str): whether to run on \"cpu\" or \"cude\" (gpu)
#         - layers (int): how many gcn layers to use
#         - intensifier_type (Literal[SuccessiveHalving, Hyperband]): what intensifier to choose. 
#         - Adj_test (Tensor) : the precomputed adjacency tensor of all graphs in the test data
#         - X_test (np.ndarray): the test features
#         - y_test (np.ndarray): the test labels
        

#     Returns: Tuple of
#     Dict[str, Any] : The hyperparameters found.
#         - epochs : int
#         - use_bias : bool
#         - use_dropout : bool
#         - dropout_prob : float
#         - learning_rate : float
#     Module : the instantiated model
#     """

#     torch_model = TorchModel(
#         model=model, 
#         Adj=Adj,
#         X=X, 
#         y=y, 
#         device=device, 
#         layers=layers, 
#         Adj_test = Adj_test,
#         X_test=X_test, 
#         y_test=y_test,
#         batch_size=1 if dataset in ["Cora", "Citeseer"] else None
#     )

#     scenario = Scenario(
#         torch_model.configspace,
#         n_trials = 2000,
#         walltime_limit = 60*60*0.5, #one hour
#         min_budget = 50,
#         max_budget = 2000 if dataset in ["Cora", "Citeseer"] else 500,
#         n_workers = 1#how many processes can run on the same volta gpu? about 4
#     )

#     #basically a bohb, run with a population of 5
#     initial_design = MultiFidelityFacade.get_initial_design(
#         scenario, 
#         n_configs=50
#     )
#     if intensifier_type=="Hyperband":
#         intensifier = Hyperband(
#             scenario, 
#             incumbent_selection = "highest budget"
#         )
#     elif intensifier_type=="SuccessHalving":
#         intensifier = SuccessiveHalving(
#             scenario, 
#             incumbent_selection = "highest budget"
#         )
#     else:
#         raise ValueError(f"The given intensifier_type is not allowed, got {intensifier_type}, but expected on of: [\"SuccessHalving\", \"Hyperband\"].")

#     smac = MultiFidelityFacade(
#         scenario, 
#         torch_model.train,
#         initial_design=initial_design,
#         intensifier=intensifier,
#         overwrite=True,
#     )

#     incumbent = smac.optimize()

#     #do the cross_validation:

#     #get the last configuration first
#     best_config = dict(incumbent)
#     #we don't optimize the epochs, we optimize early stopping to optimize that.
#     plot_trajectory([smac])

#     return (best_config | {"epochs":2000 if dataset in ["Cora", "Citeseer"] else 500}), torch_model
    
# def plot_trajectory(facades: list[AbstractFacade]) -> None:
#     """Plots the trajectory (incumbents) of the optimization process."""
#     plt.figure()
#     plt.title("Trajectory")
#     plt.xlabel("Wallclock time [s]")
#     plt.ylabel(facades[0].scenario.objectives)
#     plt.ylim(0, 0.7)

#     for facade in facades:
#         X, Y = [], []
#         for item in facade.intensifier.trajectory:
#             # Single-objective optimization
#             assert len(item.config_ids) == 1
#             assert len(item.costs) == 1

#             y = item.costs[0]
#             x = item.walltime

#             X.append(x)
#             Y.append(y)

#         plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
#         plt.scatter(X, Y, marker="x")

#     plt.legend()
#     plt.savefig("out/optimisation.png")
#     plt.close()

def unique_file(basename:str)->str:
    """Return a unique file name by appending a number to the file name if the file already exists.
        Linear runtime in the number of files with the same basename.
    """
    basename, ext = os.path.splitext(basename)
    ext = ext[1:] # remove the dot
    actualname = "%s.%s" % (basename, ext) if ext!="" else "%s" % basename
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s_(%d).%s" % (basename, next(c), ext) if ext!="" else "%s_(%d)" % (basename, next(c))
    return actualname

# def accuracy_sum(y_pred: Tensor, y_true: Tensor, max_n_nodes: int, length: int, model_type: str) -> float:
#     '''Computes the accuracy while ignoring the node padding, the classifier may behave arbitrarily there.
    
#     '''
#     coincidence_tensor = (y_pred.argmax(-1) == y_true.argmax(-1))

#     empty_node_guard_tensor = torch.tensor([[(y_true[graph][node] != 0).type(torch.float).max() for node in range(max_n_nodes)] for graph in range(length)]).type(torch.bool)

#     coincidence_tensor = (empty_node_guard_tensor & coincidence_tensor)

#     return coincidence_tensor.type(torch.float).sum().item()
