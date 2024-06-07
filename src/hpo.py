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

#fix json-save problem for numpy types
class Hyperband(Hyperband_):
    
    def save(self, filename: str | Path = "intensifier.json") -> None:
        """Saves the current state of the intensifier. In addition to the state (retrieved by ``get_state``), this
        method also saves the incumbents and trajectory.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        assert str(filename).endswith(".json")
        filename.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "incumbent_ids": [self.runhistory.get_config_id(config) for config in self._incumbents],
            "rejected_config_ids": self._rejected_config_ids,
            "incumbents_changed": self._incumbents_changed,
            "trajectory": [dataclasses.asdict(item) for item in self._trajectory],
            "state": self.get_state(),
        }
        #### THIS IS THE FIX ####
        class numpy_Encoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "item") and callable(obj.item): #numpy types
                    return obj.item()
                return json.JSONEncoder.default(self, obj)
            
        with open(filename, "w") as fp:
            json.dump(data, fp, indent=2, cls=numpy_Encoder)

from .layer import activation_function

class optObject:
    def __init__(self, device):
        self.device = th.device("cuda")
        self.model = GNN

    @property
    def configspace(self)->ConfigurationSpace:
        """The ConfigSpace to search in.
        """
        param_space:ConfigurationSpace = ConfigurationSpace()
        ####hyperparameters

        #layer & dimensions
        n_GNN_layers = Integer(name="n_GNN_layers", bounds=(1,20), default=5)
        dim_between = Integer(name="dim_between", bounds=(30, 500), default=100, log=True) #bounds=(3, 9)
        dim_M = Integer(name="dim_M", bounds=(30, 500), default=100, log=True)#bounds=(3, 15)
        dim_U = Integer(name="dim_U", bounds=(30, 500), default=100, log=True)#bounds=(3, 15)
        n_MLP_layers = Integer(name="n_MLP_layers", bounds=(1, 3), default=1)
        dim_MLP = Integer(name="dim_MLP", bounds=(10, 100), default=20, log=True) #bounds=(3, 8)
        # dim_MLP = Categorical(name="dim_MLP", items=[3], default=3) 
        n_M_layers = Integer(name="n_M_layers", bounds=(1, 3), default=1)
        n_U_layers = Integer(name="n_U_layers", bounds=(1, 3), default=1)

        #nodes, training regularization
        use_virtual_nodes = Categorical(name="use_virtual_nodes", items=[0,1], default=1) #items=[1, 0]
        n_virtual_layers = Integer(name="n_virtual_layers", bounds=(1, 3), default=1)
        use_skip = Categorical(name="use_skip", items=[0,1], default=1)
        use_residual = Categorical(name="use_residual", items=[0,1], default=1)
        dropout_prob = Float(name="dropout_prob", bounds=(0.0, 0.5), default=0.5)
        use_dropout = Categorical(name="use_dropout", items=[1, 0], default=1)
        
        #nonlinearities
        mlp_nlin = Categorical(name="mlp_nlin", items=list(activation_function.keys()), default='relu')
        m_nlin = Categorical(name="m_nlin", items=list(activation_function.keys()), default='leaky_relu')
        u_nlin = Categorical(name="u_nlin", items=list(activation_function.keys()), default='relu')

        #other training related
        batch_size = Integer(name="batch_size", bounds=(10, 1000), default=10, log=True)
        n_epochs = Integer(name="n_epochs", bounds=(20, 100), default=30) # bounds=(20,100)
        lr = Float(name="lr", bounds=(0.00001, 0.01), default=0.001, log=True) #bounds=(1e-05, 1e-02)
        lrsched = Categorical(name="lrsched", items=["cosine", "cyclic"], default="cosine")
        beta1 = Float(name="beta1", bounds=(0.9, 0.95), default=0.9)
        beta2 = Categorical(name="beta2", items=[0.999], default=0.999)
        weight_decay = Float(name="weight_decay", bounds=(1e-06, 1e-03), default=1e-05, log=True)
        use_weight_decay = Categorical(name="use_weight_decay", items=[1, 0], default=0)

        #scatter operation
        scatter_type = Categorical(name = "scatter_type", items=['sum', 'mean', 'max'], default='mean')

        #bounding conditions, st make the spaces more explicit, so it finds easier, st necessary
        use_wd = EqualsCondition(child=weight_decay, parent=use_weight_decay, value=True)
        # use_vn = GreaterThanCondition(child=use_virtual_nodes, parent=n_GNN_layers, value=1)
        use_vn = EqualsCondition(child=use_virtual_nodes, parent=n_GNN_layers, value=2)
        use_vn2 = EqualsCondition(child=n_virtual_layers, parent=use_virtual_nodes, value=1)
        use_du = GreaterThanCondition(child=dim_U, parent=n_U_layers, value=1)
        use_dp = EqualsCondition(child=dropout_prob, parent=use_dropout, value=1)
        param_space.add(n_GNN_layers, dim_between, dim_M, dim_U, n_M_layers, n_U_layers, use_virtual_nodes, n_virtual_layers, n_MLP_layers, dim_MLP, batch_size, n_epochs, lr, beta1, beta2, weight_decay, use_weight_decay, scatter_type, use_skip,lrsched, mlp_nlin, m_nlin, u_nlin, use_residual, dropout_prob, use_dropout,
            use_wd, use_vn, use_vn2, use_du, use_dp)

        return param_space

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
            config["dim_MLP"], 
            config["n_virtual_layers"], 
            config["n_GNN_layers"], 
            config["dim_between"],
            config["dim_M"], 
            config["dim_U"], 
            config["n_M_layers"], 
            config["n_U_layers"],
            mlp_nonlin=config["mlp_nlin"],
            m_nonlin=config["m_nlin"],
            u_nonlin=config["u_nlin"],
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

def hpt(device) -> Dict[str, Any]:
    """
    Hyperparameter optimization for the GNN model using the ZINC dataset.
    
    Args:
        device: The device to use for torch computations.
    
    Returns:
        List of dicts that hold the best hyperparameters found for each dataset.
    """
    gnn_opt = optObject(device)

    # setup the hyperparameter optimization scenario
    scenario = Scenario(
        configspace=gnn_opt.configspace,
        n_trials = 300,
        n_workers=1,
        trial_walltime_limit=30*60,
        min_budget = 1000,
        max_budget = 10000,
    )

    intensifier = Hyperband(scenario, incumbent_selection = "highest budget")

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs_per_hyperparamter=4)

    # setup the hyperparameter optimization facade
    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=gnn_opt.objective,
        intensifier=intensifier,
        initial_design=initial_design,
        logging_level=20,#INFO
        overwrite=True
    )

    # run the hyperparameter optimization
    incumbent = smac.optimize()
    if isinstance(incumbent, list):
        incumbent = incumbent[0]
    return dict(incumbent)





