from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


class Node2VecModel(nn.Module):
    """
    Implements a Node2Vec model using PyTorch's neural network module. This model learns low-dimensional 
    embeddings for nodes in a graph based on their neighborhood. It is designed to be trained using the
    Skip-Gram architecture, where the model predicts context nodes (positive samples) and non-context nodes
    (negative samples) given a target node.

    Attributes:
        embeddings (nn.Embedding): An embedding layer that maps node indices to embedding vectors.

    Args:
        num_nodes (int): The total number of nodes in the graph. This determines the size of the embedding layer.
        embedding_dim (int): The dimensionality of the embedding vectors.

    Methods:
        forward(start_node, walk, negative_samples):
            Performs a forward pass of the Node2Vec model using the input data.

        loss(positive_probs, negative_probs):
            Computes the loss of the model using the binary cross-entropy loss function.
    """

    embeddings: nn.Embedding

    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        """
        Initializes the Node2VecModel instance by setting up the embedding layer.
        """
        super(Node2VecModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    from typing import Tuple

    def forward(self, start_node: torch.Tensor, walk: torch.Tensor, negative_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Node2Vec model. Computes embeddings for start nodes, walk nodes, and negative samples.

        Args:
            start_node (torch.Tensor): A tensor containing indices of start nodes.
            walk (torch.Tensor): A tensor containing indices of walk nodes that are typically neighbors of the start nodes.
            negative_samples (torch.Tensor): A tensor containing indices of negative samples which are not neighbors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors with probabilities of positive and negative samples.
        """
        embedded_start = self.embeddings(start_node).unsqueeze(1)
        embedded_walk = self.embeddings(walk)
        embedded_neg_samples = self.embeddings(negative_samples)

        positive_scores = torch.sum(embedded_start * embedded_walk, dim=-1)
        negative_scores = torch.sum(
            embedded_start * embedded_neg_samples, dim=-1)

        positive_probs = torch.sigmoid(positive_scores)
        negative_probs = torch.sigmoid(-negative_scores)

        return positive_probs, negative_probs

    def loss(self, positive_probs: torch.Tensor, negative_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the model's loss using binary cross-entropy.

        Args:
            positive_probs (torch.Tensor): Probabilities associated with positive samples.
            negative_probs (torch.Tensor): Probabilities associated with negative samples.

        Returns:
            torch.Tensor: The computed loss value, as a single tensor.
        """
        positive_loss = -torch.log(positive_probs + 1e-10).mean()
        negative_loss = -torch.log(1 - negative_probs + 1e-10).mean()
        return positive_loss + negative_loss
