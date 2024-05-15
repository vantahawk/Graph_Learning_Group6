from sklearn.model_selection import KFold
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
        support = torch.bmm(adj, x)
        output = torch.bmm(support, self.weight)
        return F.relu(output)


class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(num_features, 64)
        self.gcn2 = GCNLayer(64, 64)
        self.gcn3 = GCNLayer(64, 64)
        self.gcn4 = GCNLayer(64, 64)
        self.gcn5 = GCNLayer(64, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, adj):
        y = self.gcn1(x, adj)
        y = self.gcn2(y, adj)
        y = self.gcn3(y, adj)
        y = self.gcn4(y, adj)
        y = self.gcn5(y, adj)

        # Sum pooling
        y = torch.sum(y, dim=1)

        # Pass through MLP
        y = self.mlp(y)
        return y
