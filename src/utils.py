from pathlib import Path
from networkx import Graph
import pickle
from typing import List,Union, Tuple,Dict, Literal, Any
import os
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import itertools

import ctypes as ct

from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Configuration, Integer, Float, Categorical
from ConfigSpace.hyperparameters import IntegerHyperparameter
from ConfigSpace.conditions import InCondition, LessThanCondition

Shape = Union[Tuple[int], Union[int, int], Union[int, int, int]]

def load_graphs_dataset(dataset_path:str|Path)->List[Graph]:
    """Loads a pickled dataset and returns it. The expected dataset is a list of networkx.Graph instances.
    
    Args:
    - dataset_path (str|Path): The path to the pkled object. If a Path object is parsed, only allows for posix-paths

    Returns:
    List[Graph] - The unpickled list of Graphs and the Labels (may be a vector or int) 
    
    The labels are set as attributes to the graph objects, access via: `graph.graph["label"]`.
    The node-labels are set as data on the nodes. Access via `graph.nodes(data='node_label')[node_id]`, other data may be accessed alike.
    """
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


def make_data(graphs:List[Graph], dataset:str)->Tuple[np.ndarray, np.ndarray]:
    
    features, labels = None, None

    match dataset:
        case "ENZYMES":
            #prepare the feature tensor: the one-hot vectors
            one_hot:np.ndarray = np.zeros((
                len(graphs), #n_graphs
                max([graph.number_of_nodes() for graph in graphs]), #n_nodes
                np.max([        #n_features
                    np.max(list(dict(graph.nodes(data="node_label")).values()))
                for graph in graphs])+1
            ))
            #set the 1 for the one-hot vector for each node
            for g in range(one_hot.shape[0]):
                for n, nname in enumerate(graphs[g].nodes):
                    one_hot[g, n, graphs[g].nodes(data="node_label")[nname]] = 1

            #now build the node_attribute tensor
            node_attrs:np.ndarray = np.zeros((one_hot.shape[0], one_hot.shape[1], 18))

            for g in range(node_attrs.shape[0]):
                for n, nname in enumerate(graphs[g].nodes):
                    attrs:np.ndarray = np.array(graphs[g].nodes(data="node_attributes")[nname])
                    node_attrs[g, n] = attrs / np.linalg.norm(attrs) #normalize by l2

            features = np.concatenate((one_hot, node_attrs), axis=2)
            #normalize features
            fmean = features.mean(axis=2, keepdims=True)
            fstd = features.std(axis=2, keepdims=True)
            fstd[np.where(fstd==0)]=1
            features = (features-fmean)/fstd
            
            max_label = max([graph.graph["label"] for graph in graphs])
            labels = np.zeros((len(graphs), max_label+1))
            for g, graph in enumerate(graphs):
                labels[g, graph.graph["label"]] = 1
                
        case "NCI1":
            #prepare the feature tensor: one-hot vectors
            features = np.zeros((
                len(graphs), #n_graphs
                max([graph.number_of_nodes() for graph in graphs]), #n_nodes
                np.max([        #n_features
                    np.max(list(dict(graph.nodes(data="node_label")).values()))
                for graph in graphs])+1
            ))
            #set the 1 for the one-hot vector for each node
            for g in range(features.shape[0]):
                for n, nname in enumerate(graphs[g].nodes):
                    features[g, n, graphs[g].nodes(data="node_label")[nname]] = 1
                    
            max_label = max([graph.graph["label"] for graph in graphs])
            labels = np.zeros((len(graphs), max_label+1))
            for g, graph in enumerate(graphs):
                labels[g, graph.graph["label"]] = 1
                
        case "Cora" | "Citeseer":
            
            features = np.ndarray((
                len(graphs), 
                max([graph.number_of_nodes() for graph in graphs]),
                len(graphs[0].nodes(data="node_attributes")[1])
            ))
            num_classes = np.max([        #n_features
                    np.max(list(dict(graph.nodes(data="node_label")).values()))
                for graph in graphs])+1
            labels = np.ndarray((
                len(graphs), 
                max([graph.number_of_nodes() for graph in graphs]),
                num_classes
            ))

            for g in range(features.shape[0]):

                for n, nname in enumerate(graphs[g].nodes):
                    features[g,n] = np.array(graphs[g].nodes(data="node_attributes")[nname])
                index_labels = torch.tensor(np.array(list(dict(graphs[g].nodes(data="node_label")).values()), dtype=int))
                # print(index_labels)
                labels[g] = F.one_hot(index_labels, num_classes=num_classes).numpy()

            #normalize features
            fmean = features.mean(axis=2, keepdims=True)
            fstd = features.std(axis=2, keepdims=True)
            fstd[np.where(fstd==0)]=0.1**8
            features = (features-fmean)/fstd


        case _:
            raise ValueError(f"0,The given dataset {dataset} cannot be converted into data.")

    return features, labels

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

class TorchModel:
    @property
    def configspace(self) -> ConfigurationSpace:

        cs = ConfigurationSpace(seed=0)

        # epochs = Integer("epochs", (50, 500), default=100)

        batch_size = Integer("batch_size", (1, 10000), default=500, log=True)

        use_bias = Integer("use_bias", (0, 1), default=0) #bool

        use_dropout = Integer("use_dropout", (0, 1), default=0) #boool
        dropout_prob = Float("dropout_prob", (0.01, 0.8), default=0.2)

        learning_rate = Float("learning_rate", (0.00001, 0.01), default=0.005, log=True)

        use_early_stop = Integer("use_early_stop", (0, 1), default=1) #bool
        es_patience = Integer("es_patience", (1, 10), default=5)
        es_min_delta = Float("es_min_delta", (0.0001, 0.1), default = 0.005, log=True)

        grad_clip = Float("grad_clip", (1.0, 5.0), default=2.0)
        weight_decay = Float("weight_decay", (10**(-6), 10**(0)), log=True)

        #if use_dropout is set, use the dropout_prob hp
        #this saves search space
        use_drop_prob = InCondition(dropout_prob, use_dropout, [True])
        # use_epochs = LessThanCondition(epochs, epochs, 0)#change if we use this as a hyperparameter and not use a budget for this
        use_es_pat = InCondition(es_patience, use_early_stop, [True])
        use_es_md = InCondition(es_min_delta, use_early_stop, [True])

        cs.add(use_bias, use_dropout, dropout_prob, learning_rate, use_early_stop, es_patience, es_min_delta, grad_clip, weight_decay)
        if not self.batch_size:
            cs.add(batch_size)
        cs.add(use_drop_prob, use_es_pat, use_es_md)

        return cs

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


    def train(self, config:Configuration, seed:int=0, budget:int=50):
        budget = int(budget)
        seed = int(seed)

        config_dict = dict(config)

        # create dataset and loader for mini batches
        train_dataset = TensorDataset(self.Adj, self.X_train, self.y_train )
        #TODO: make batch_size depend on 
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size if self.batch_size>0 else config_dict["batch_size"], shuffle=True)

        # construct neural network and move it to device
        model:Module = self.model(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            hidden_dim=64, 
            num_layers=self.layers, 
            use_bias=config_dict["use_bias"], 
            use_dropout=config_dict["use_dropout"], 
            dropout_prob=config_dict.get("dropout_prob", 0.0)
        )

        model.train()
        model.to(self.device)

        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=config_dict["learning_rate"], weight_decay=config_dict["weight_decay"])


        early_stop = ValidationLossEarlyStopping(patience=config_dict["es_patience"], min_delta=config_dict["es_min_delta"]) if config_dict["use_early_stop"] else None

        self.Adj_test = self.Adj_test.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)


        train_loss:Tensor = None
        val_loss:Tensor = None
        for epoch in range(budget):
            
            for adj, x, y_true in train_loader:#batches
                # set gradients to zero
                opt.zero_grad()

                # move data to device
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                adj = adj.to(self.device)

                # forward pass and loss
                y_pred:Tensor = model(adj, x)
                
                if self.dataset in ["Cora", "Citeseer"]:
                    train_loss = F.cross_entropy(y_pred[0], y_true[0])
                else:
                    train_loss = F.cross_entropy(y_pred, y_true)

                # if train_loss.item() > 1.0:
                #     #if our loss is weird, throw it away, somehow the -log must have gotten small values?
                #     continue
                
                # backward pass and sgd step
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config_dict["grad_clip"], error_if_nonfinite=True)
                opt.step()

             #we just use the valid_set here for early stopping
            with torch.no_grad():
                y_test_pred:Tensor = model(self.Adj_test, self.X_test)
                if self.dataset in ["Cora", "Citeseer"]:
                    val_loss = F.cross_entropy(y_test_pred[0], self.y_test[0])
                else:
                    val_loss = F.cross_entropy(y_test_pred, self.y_test)

                # acc= (y_test_pred == self.y_test).float().mean().item()
                # print("Current Accuracy:", acc)

            if config_dict["use_early_stop"] and early_stop(val_loss.item()):
                break

        #use the final validation loss as decision factor when comparing configurations
        print("Solved an smac opt step, got val_loss:", val_loss.item())
        return val_loss.item()

def opt_with_smac(model:Module, Adj:Tensor, X:np.ndarray, y:np.ndarray, dataset:str, device:str, layers:int, intensifier_type:Literal["SuccessHalving", "Hyperband"]="SuccessiveHalving", Adj_test:Tensor=None, X_test:np.ndarray=None, y_test:np.ndarray=None)->Tuple[Dict[str, Any], TorchModel]:
    """#TODO Write this docstring
    
    Args:
        - model (Module) : a pytorch module object, not an instance!
        - Adj (Tensor): the precomputed adjacency tensor of all graphs in the data
        - X (np.ndarray): the features, if X_test the train features
        - y (np.ndarray): the labels, if y_test the train labels#
        - dataset (str): the dataset used
        - device (str): whether to run on \"cpu\" or \"cude\" (gpu)
        - layers (int): how many gcn layers to use
        - intensifier_type (Literal[SuccessiveHalving, Hyperband]): what intensifier to choose. 
        - Adj_test (Tensor) : the precomputed adjacency tensor of all graphs in the test data
        - X_test (np.ndarray): the test features
        - y_test (np.ndarray): the test labels
        

    Returns: Tuple of
    Dict[str, Any] : The hyperparameters found.
        - epochs : int
        - use_bias : bool
        - use_dropout : bool
        - dropout_prob : float
        - learning_rate : float
    Module : the instantiated model
    """

    torch_model = TorchModel(
        model=model, 
        Adj=Adj,
        X=X, 
        y=y, 
        device=device, 
        layers=layers, 
        Adj_test = Adj_test,
        X_test=X_test, 
        y_test=y_test,
        batch_size=1 if dataset in ["Cora", "Citeseer"] else None
    )

    scenario = Scenario(
        torch_model.configspace,
        n_trials = 2000,
        walltime_limit = 60*60*1, #one hour
        min_budget = 50,
        max_budget = 10000 if dataset in ["Cora", "Citeseer"] else 500,
        n_workers = 1#how many processes can run on the same volta gpu? about 4
    )

    #basically a bohb, run with a population of 5
    initial_design = MultiFidelityFacade.get_initial_design(
        scenario, 
        n_configs=5
    )
    if intensifier_type=="Hyperband":
        intensifier = Hyperband(
            scenario, 
            incumbent_selection = "highest budget"
        )
    elif intensifier_type=="SuccessHalving":
        intensifier = SuccessiveHalving(
            scenario, 
            incumbent_selection = "highest budget"
        )
    else:
        raise ValueError(f"The given intensifier_type is not allowed, got {intensifier_type}, but expected on of: [\"SuccessHalving\", \"Hyperband\"].")

    smac = MultiFidelityFacade(
        scenario, 
        torch_model.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    #do the cross_validation:

    #get the last configuration first
    best_config = dict(incumbent)
    #we don't optimize the epochs, we optimize early stopping to optimize that.
    plot_trajectory([smac])

    return (best_config | {"epochs":10000 if dataset in ["Cora", "Citeseer"] else 500}), torch_model
    
def plot_trajectory(facades: list[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.7)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.savefig("out/optimisation.png")
    plt.close()

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
