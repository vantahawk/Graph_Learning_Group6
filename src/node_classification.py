from typing import Tuple
import pickle
from networkx import Graph
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from node_2_vec import Node2VecModel
from random_walker import RandomWalkDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List


def generate_embeddings(G: Graph, p: float, q: float, embedding_dim: int = 128, num_epochs: int = 100) -> torch.Tensor:
    """
    Generates embeddings for nodes in a given graph using the Node2Vec algorithm.

    Args:
        G (Graph): The graph for which embeddings are to be generated.
        p (float): The return parameter in the random walk (controls the likelihood of revisiting a node in the walk).
        q (float): The in-out parameter in the random walk (controls the exploration-exploitation balance).
        embedding_dim (int): The number of dimensions for the node embeddings.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        torch.Tensor: The embeddings matrix, where each row corresponds to a node in the graph.
    """
    max_node_index = max(G.nodes())
    model = Node2VecModel(max_node_index + 1, embedding_dim)
    dataset = RandomWalkDataset(G, p=p, q=q, l=5, l_ns=5)
    loader = DataLoader(dataset, batch_size=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for start_node, walk, neg_samples in loader:
            optimizer.zero_grad()
            positive_probs, negative_probs = model(
                start_node, walk, neg_samples)
            loss = model.loss(positive_probs, negative_probs)
            loss.backward()
            optimizer.step()

    return model.embeddings.weight.data


def evaluate_embeddings(embeddings: torch.Tensor, nodes: list) -> Tuple[float, float]:
    """
    Evaluates the embeddings by training a logistic regression model and performing cross-validation.

    Args:
        embeddings (torch.Tensor): Node embeddings to be evaluated.
        nodes (list): The list of node indices corresponding to the embeddings.

    Returns:
        tuple[float, float]: The mean accuracy and standard deviation of the logistic regression model.
    """
    clf = LogisticRegression(max_iter=1000)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, nodes, cv=kf)
    return np.mean(scores), np.std(scores)


with open('datasets/Citeseer/data.pkl', 'rb') as data:
    graphs: List[Graph] = pickle.load(data)
    for G in graphs:
        params = [(1, 1), (0.1, 1), (1, 0.1)]
        results = {}
        for p, q in params:
            embeddings = generate_embeddings(G, p, q)
            mean_acc, std_dev = evaluate_embeddings(
                embeddings, list(G.nodes()))
            results[(p, q)] = (mean_acc, std_dev)
            print(
                f"Results for p={p}, q={q}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_dev:.4f}")
        best_params = max(results, key=results.get)
        print(
            f"Best parameters for graph: p={best_params[0]}, q={best_params[1]}")
        print()

with open('datasets/Cora/data.pkl', 'rb') as data:
    graphs: List[Graph] = pickle.load(data)
    for G in graphs:
        params = [(1, 1), (0.1, 1), (1, 0.1)]
        results = {}
        for p, q in params:
            embeddings = generate_embeddings(G, p, q)
            mean_acc, std_dev = evaluate_embeddings(
                embeddings, list(G.nodes()))
            results[(p, q)] = (mean_acc, std_dev)
            print(
                f"Results for p={p}, q={q}: Mean Accuracy = {mean_acc:.4f}, Std Dev = {std_dev:.4f}")
        best_params = max(results, key=results.get)
        print(
            f"Best parameters for graph: p={best_params[0]}, q={best_params[1]}")
        print()
