from torch import Module, Tensor
from torch.nn import Sequential, Linear,ReLU, functional as F, AvgPool1d, Dropout
from .layers import GeneralGCNLayer, PoolNodeEmbeddings, Padding
from typing import Literal

class GraphLevelGCN(Module):

    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int, num_layers:int, use_bias:bool=False, use_dropout:bool=False, dropout_prob:float=0.5):
        """
        A graph-level classifier model.
        
        Args: 
            - input_dim (int): The initial graph embedding dimension
            - output_dim (int): The number of labels
            - hidden_dim (int): The size of the hidden layers.
            - num_layers (int): the number of layers for the GCN in total
            
            - use_bias (bool): whether to use a bias in the GCN
            - use_dropout (bool): wether to use dropout layers
        """

        super().__init__()

        self.num_layers = num_layers

        self.input_gcn_layer = GeneralGCNLayer(input_dim, hidden_dim, use_bias=use_bias)
        self.output_gcn_layer = GeneralGCNLayer(hidden_dim, hidden_dim, use_bias=use_bias)

        self.hidden_layers = [
            GeneralGCNLayer(hidden_dim, hidden_dim, use_bias=use_bias) for _ in range(num_layers-2)
        ]

        if use_dropout:#inject one dropout layer before every hidden layer.
            num_hid_layers = len(self.hidden_layers)
            for i in range(num_hid_layers):
                self.hidden_layers.insert(2*i, Dropout(p=dropout_prob))

        self.pool_layer = PoolNodeEmbeddings(type="sum")#size is  1 x hidden_dim

        #TODO: possible do a skip connection from the input features into the classifier

        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, h_0:Tensor):
        """The forward pass of the graph classification model

        Args:
            - h_0 (Tensor): Size batch_size x n_nodes x input_dim
        """
        h = self.input_gcn_layer(h_0)
        h = F.relu(h)

        for i in range(self.num_layers -2):
            h = self.hidden_layers[i](h)
            h = F.relu(h)

        h = self.output_gcn_layer(h)
        h = F.relu(h)

        h = self.pool_layer(h)
        y = self.classifier(h)

        return y


class NodeLevelGCN(Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int, num_layers:int, use_bias:bool=False, use_dropout:bool=False, dropout_prob:float=0.5):
        """
        A node-level classifier model.
        
        Args: 
            - input_dim (int): The initial graph embedding dimension
            - output_dim (int): The number of labels.
            - hidden_dim (int): The size of the hidden layers.
            - num_layers (int): the number of layers for the GCN in total
            - use_bias (bool): whether to use a bias in the GCN
            - use_dropout (bool): Whether to apply dropout
            - dropout_prob (bool): The dropout probability
        """

        super().__init__()

        self.num_layers = num_layers

        self.input_gcn_layer = GeneralGCNLayer(input_dim, hidden_dim)
        self.output_gcn_layer = GeneralGCNLayer(hidden_dim, hidden_dim)

        self.hidden_layers = [
            GeneralGCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers-2)
        ]
        if use_dropout:#inject one dropout layer before every hidden layer.
            num_hid_layers = len(self.hidden_layers)
            for i in range(num_hid_layers):
                self.hidden_layers.insert(2*i, Dropout(p=dropout_prob))

        self.classifier_layer = Linear(hidden_dim, output_dim)

    def forward(self, h_0:Tensor):
        """The forward pass of the graph classification model

        Args:
            - h_0 (Tensor): Size batch_size x n_nodes x input_dim
        """
        h = self.input_gcn_layer(h_0)
        h = F.relu(h)

        for i in range(self.num_layers -2):
            h = self.hidden_layers[i](h)
            h = F.relu(h)

        h = self.output_gcn_layer(h)

        y = self.classifier_layer(h)

        return y
