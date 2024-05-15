import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(
            in_features, out_features) * np.sqrt(2. / in_features))

    def forward(self, x, adj):
        # Batch matrix multiplication between the normalized adjacency matrix and the input features
        support = torch.bmm(adj, x)
        # Batch matrix multiplication between the support and the weight matrix
        output = torch.bmm(support, self.weight)
        # Apply the ReLU activation function
        output = F.relu(output)
        return output
