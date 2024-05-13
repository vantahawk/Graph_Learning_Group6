from pathlib import Path
from networkx import Graph
import pickle
from typing import List,Union, Tuple,Dict, Literal, Any
import os
import numpy as np
from torch import Module, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

from smac import MultiFidelityFacade, Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.hyperparameters

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
                    np.max(
                        dict(graph.nodes(data="node_label")).values())
                for graph in graphs])
            ))
            #set the 1 for the one-hot vector for each node
            for g in one_hot.shape[0]:
                for n in one_hot.shape[1]:
                    one_hot[g, n, graphs[g].nodes(data="node_label")[n+1]] = 1

            #now build the node_attribute tensor
            node_attrs:np.ndarray = np.zeros((one_hot.shape[0], one_hot.shape[1], 18))

            for g in node_attrs.shape[0]:
                for n in range(graphs[g].number_of_nodes()):
                    attrs:np.ndarray = np.array(graphs[g].nodes(data="node_attributes")[n+1])
                    node_attrs[g, n] = attrs / np.linalg.norm(attrs)

            features = np.concatenate((one_hot, node_attrs), axis=2)
            
            labels = np.array([graph.graph["label"] for graph in graphs])
        case "NCI1":
            #prepare the feature tensor: one-hot vectors
            features = np.zeros((
                len(graphs), #n_graphs
                max([graph.number_of_nodes() for graph in graphs]), #n_nodes
                np.max([        #n_features
                    np.max(
                        dict(graph.nodes(data="node_label")).values())
                for graph in graphs])
            ))
            #set the 1 for the one-hot vector for each node
            for g in features.shape[0]:
                for n in features.shape[1]:
                    features[g, n, graphs[g].nodes(data="node_label")[n+1]] = 1

            labels = np.array([graph.graph["label"] for graph in graphs])
        case "Cora" | "Citeseer":
            
            features = np.ndarray((
                len(graphs), 
                max([graph.number_of_nodes() for graph in graphs]),
                len(graphs[0].nodes(data="node_attributes")[1])
            ))
            labels = np.ndarray((
                len(graphs), 
                max([graph.number_of_nodes() for graph in graphs]),
            ))

            for g in range(features.shape[0]):

                for n in range(features.shape[1]):
                    features[g,n] = np.array(graphs[g].nodes(data="node_attributes")[n])

                labels[g] = np.array(list(dict(graphs[g].nodes(data="node_label")).values()))

        case _:
            raise ValueError(f"The given dataset {dataset} cannot be converted into data.")

    return features, labels

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
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

        epochs = Integer("epochs", (50, 500), default=100)

        if(self.batch_size):
            batch_size = Integer("batch_size", (batch_size, batch_size), default=batch_size)
        else:
            batch_size = Integer("batch_size", (1, 200), default=50)

        use_bias = Categorical("use_bias", [False, True], default=False)

        use_dropout = Categorical("use_dropout", [False, True], default=False)
        dropout_prob = Float("dropout_prob", (0.01, 0.8), default=0.2)

        learning_rate = Float("learning_rate", (0.00001, 0.01), default=0.001, log=True)

        use_early_stop = Categorical("use_early_stop", [False, True], default=True)
        es_patience = Integer("es_patience", (0, 10), default=2)
        es_min_delta = Float("es_min_delta", (0.0, 0.1), default = 0.01)

        #if use_dropout is set, use the dropout_prob hp
        #this saves search space
        use_drop_prob = InCondition(dropout_prob, use_dropout, [True])
        use_epochs = LessThanCondition(epochs, epochs, 0)#change if we use this as a hyperparameter and not use a budget for this
        use_es_pat = InCondition(es_patience, use_early_stop, [True])
        use_es_md = InCondition(es_min_delta, use_early_stop, [True])

        cs.add_hyperparameters([epochs, use_bias, use_dropout, dropout_prob, learning_rate, use_early_stop, es_patience, es_min_delta])
        cs.add_conditions([use_drop_prob, use_epochs, use_es_pat, use_es_md])

        return cs

    def __init__(self, model:Module, X:np.ndarray, y:np.ndarray, device:str, layers:int, batch_size:int=None, X_test:np.ndarray=None, y_test:np.ndarray=None):

        self.model:Module = model
        self.X_train:Tensor = Tensor(X) if X_test else Tensor(X[:int(0.9*X.shape[0])]) #else do a train_test split
        self.y_train:Tensor = Tensor(y) if y_test else Tensor(y[:int(0.9*y.shape[0])])#else do a train_test split

        self.X_test = X_test if X_test else Tensor(X[:int(0.9*X.shape[0])])
        """10\% datasplit"""
        self.y_test = y_test if y_test else Tensor(y[int(0.9*y.shape[0]):])
        """10\% datasplit"""

        # determine input and output dimensions
        self.input_dim:int = self.X_train.shape[2]
        self.output_dim:int = self.y_train.shape[1]

        self.device:str = device
        self.layers:int = layers
        self.batch_size:int = batch_size

    def train(self, config:Configuration, seed:int=0, budget:int=50):

        config_dict = config.get_dictionary()

        # create dataset and loader for mini batches
        train_dataset = TensorDataset(self.X_train, self.y_train )
        #TODO: make batch_size depend on 
        train_loader = DataLoader(train_dataset, batch_size=config_dict["batch_size"], shuffle=True)

        # construct neural network and move it to device
        model = self.model(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            hidden_dim=64, 
            num_layers=self.layers, 
            use_bias=config_dict["use_bias"], 
            use_dropout=config_dict["use_dropout"], 
            dropout_prob=config_dict["dropout_prob"]
        )

        model.train()
        model.to(self.device)

        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=config_dict["learning_rate"])

        early_stop = ValidationLossEarlyStopping(patience=config_dict["es_patience"], min_delta=config_dict["es_min_delta"])

        train_loss:Tensor = None
        val_loss:Tensor = None
        for epoch in range(budget):
            
            for x, y_true in train_loader:#batches
                # set gradients to zero
                opt.zero_grad()

                # move data to device
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # forward pass and loss
                y_pred:Tensor = model(x)
                train_loss = F.cross_entropy(y_pred, y_true)

                #we just use the valid_set here for early stopping
                with torch.no_grad():
                    y_test_pred = model(self.X_test)
                    val_loss = F.cross_entropy(y_test_pred, self.y_test)

                if config_dict["use_early_stop"] and early_stop(val_loss.item()):
                    break

                # backward pass and sgd step
                train_loss.backward()
                opt.step()
        

        return val_loss.item()

def opt_with_smac(model:Module, X:np.ndarray, y:np.ndarray, dataset:str, device:str, layers:int, intensifier_type:Literal["SuccessHalving", "Hyperband"]="SuccessiveHalving", X_test:np.ndarray=None, y_test:np.ndarray=None)->Tuple[Dict[str, Any], TorchModel]:
    """#TODO Write this docstring
    
    Args:
        - model (Module) : a pytorch module object, not an instance!
        - X (np.ndarray): the features, if X_test the train features
        - y (np.ndarray): the labels, if y_test the train labels#
        - dataset (str): the dataset used
        - device (str): whether to run on \"cpu\" or \"cude\" (gpu)
        - layers (int): how many gcn layers to use
        - intensifier_type (Literal[SuccessiveHalving, Hyperband]): what intensifier to choose. 
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
        X=X, 
        y=y, 
        device=device, 
        layers=layers, 
        X_test=X_test, 
        y_test=y_test,
    )

    scenario = Scenario(
        torch_model.configspace,
        n_trials = 500,
        walltime_limit = 60*30, #half an hour
        min_budget = 20,
        max_budgt = 400,
        n_workers = 1#how many cpus/graphics cards we have?
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
    best_config = incumbent.get_dictionary()
    last_budget = incumbent.ta_runs[-1].budget

    return (best_config | {"epochs":last_budget}), torch_model
    
