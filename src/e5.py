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


if __name__ == '__main__':

    from torch.utils.data import TensorDataset, DataLoader

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.adj)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            with torch.no_grad():
                pred = model(data.x, data.adj).max(1)[1]
                correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    # Load dataset
    data = np.load('datasets/NCI1/data.pkl', allow_pickle=True)
    data = torch.tensor(data)
    dataset = TensorDataset(data)
    num_classes = len(data[0][1])
    num_features = data.shape[1]

    # Cross-validation
    kfold = KFold(10, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = GCN(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(50):
            train_loss = train()
            test_acc = test(test_loader)
            print(
                f"Fold {fold}, Epoch {epoch}, Loss: {train_loss}, Test Acc: {test_acc}")
