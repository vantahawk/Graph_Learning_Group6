import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from node_2_vec import train_node2vec
from random_walker import RW_Iterable
import random
import pickle
from typing import List, Tuple, Dict


def create_train_test_split(graph: nx.Graph, test_size: float = 0.2, seed: int = 42) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Splits the edges of a graph into training and evaluation sets while ensuring the training graph remains connected.

    Args:
        graph (nx.Graph): The input graph.
        test_size (float): The proportion of edges to be used as evaluation data.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training edges and evaluation edges.
    """
    random.seed(seed)
    edges = list(graph.edges())
    num_test = int(len(edges) * test_size)

    # Get a spanning tree to ensure the training graph is connected
    spanning_tree = list(nx.bfs_edges(
        graph, source=list(graph.nodes(data=True))[0][0]))
    remaining_edges = list(set(edges) - set(spanning_tree))

    random.shuffle(remaining_edges)
    eval_edges = remaining_edges[:num_test]
    train_edges = remaining_edges[num_test:] + spanning_tree

    return train_edges, eval_edges


def sample_negative_edges(graph: nx.Graph, num_samples: int, seed: int = 42) -> List[Tuple[int, int]]:
    """
    Samples negative edges (non-existent edges) from a graph.

    Args:
        graph (nx.Graph): The input graph.
        num_samples (int): The number of negative edges to sample.
        seed (int): The random seed for reproducibility.

    Returns:
        list: A list of negative edges.
    """
    random.seed(seed)
    nodes = list(graph.nodes(data=True))
    neg_edges = set()

    while len(neg_edges) < num_samples:
        u, v = random.sample(nodes, 2)
        if not graph.has_edge(u[0], v[0]):
            neg_edges.add((u[0], v[0]))

    return list(neg_edges)


def generate_edge_embeddings(embeddings: np.ndarray, edges: List[Tuple[int, int]], node_mapping: Dict[int, int]) -> np.ndarray:
    """
    Generates edge embeddings using the Hadamard product of node embeddings.

    Args:
        embeddings (np.ndarray): The node embeddings.
        edges (list): The list of edges.
        node_mapping (dict): A mapping from original node IDs to embedding indices.

    Returns:
        np.ndarray: The edge embeddings.
    """
    edge_embeddings = []
    for u, v in edges:
        mapped_u = node_mapping[u]
        mapped_v = node_mapping[v]
        edge_emb = np.multiply(embeddings[mapped_u], embeddings[mapped_v])
        edge_embeddings.append(edge_emb)
    return np.array(edge_embeddings)


def link_prediction(graph: nx.Graph, dim: int, p: float, q: float, l: int, l_ns: int, n_batches: int, batch_size: int, device: str, test_size: float = 0.2, seed: int = 42) -> Tuple[float, float]:
    """
    Performs link prediction on a graph using Node2Vec and logistic regression.

    Args:
        graph (nx.Graph): The input graph.
        dim (int): The dimension of the node embeddings.
        p (float): The return parameter for Node2Vec.
        q (float): The in-out parameter for Node2Vec.
        l (int): The length of each random walk.
        l_ns (int): The number of negative samples per node.
        n_batches (int): The number of batches for training Node2Vec.
        batch_size (int): The size of each batch.
        device (str): The device to use for training ('cpu' or 'cuda').
        test_size (float): The proportion of edges to be used as evaluation data.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: The accuracy and ROC-AUC of the link prediction model.
    """
    # Create a node mapping
    node_mapping = {node[0]: i for i,
                    node in enumerate(graph.nodes(data=True))}

    train_edges, eval_edges = create_train_test_split(graph, test_size, seed)
    num_eval_edges = len(eval_edges)

    train_neg_edges = sample_negative_edges(graph, len(train_edges), seed)
    eval_neg_edges = sample_negative_edges(graph, num_eval_edges, seed)

    # Create the training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(graph.nodes(data=True))
    G_train.add_edges_from(train_edges)

    # Node2Vec Embedding
    dataset = RW_Iterable(G_train, p, q, l, l_ns,
                          batch_size, set_node_labels=False)
    embeddings = train_node2vec(dataset, dim, l, n_batches, batch_size, device)

    # Generate edge embeddings
    pos_train_edge_emb = generate_edge_embeddings(
        embeddings, train_edges, node_mapping)
    neg_train_edge_emb = generate_edge_embeddings(
        embeddings, train_neg_edges, node_mapping)
    pos_eval_edge_emb = generate_edge_embeddings(
        embeddings, eval_edges, node_mapping)
    neg_eval_edge_emb = generate_edge_embeddings(
        embeddings, eval_neg_edges, node_mapping)

    # Training data
    X_train = np.vstack((pos_train_edge_emb, neg_train_edge_emb))
    y_train = np.hstack((np.ones(len(pos_train_edge_emb)),
                        np.zeros(len(neg_train_edge_emb))))

    # Evaluation data
    X_eval = np.vstack((pos_eval_edge_emb, neg_eval_edge_emb))
    y_eval = np.hstack((np.ones(len(pos_eval_edge_emb)),
                       np.zeros(len(neg_eval_edge_emb))))

    # Train logistic regression classifier
    classifier = LogisticRegression(max_iter=10000, n_jobs=-1)
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_eval)
    y_proba = classifier.predict_proba(X_eval)[:, 1]

    accuracy = accuracy_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_proba)

    return accuracy, roc_auc


if __name__ == "__main__":
    # Define parameters
    dim = 128  # Embedding dimension
    p = 1.0  # Return parameter
    q = 1.0  # In-out parameter
    l = 10  # Length of random walk
    l_ns = 5  # Number of negative samples
    n_batches = 100  # Number of batches for training
    batch_size = 32  # Batch size
    device = 'cpu'  # Training device ('cpu' or 'cuda')

    with open('datasets/Facebook/data.pkl', 'rb') as data:
        G_facebook = pickle.load(data)[0]

    with open('datasets/PPI/data.pkl', 'rb') as data:
        G_ppi = pickle.load(data)[0]

    # Run link prediction on Facebook graph
    fb_accuracy, fb_roc_auc = link_prediction(
        G_facebook, dim, p, q, l, l_ns, n_batches, batch_size, device)
    print(f'Dataset: Facebook, Accuracy: {fb_accuracy}, ROC-AUC: {fb_roc_auc}')

    # Run link prediction on PPI graph
    ppi_accuracy, ppi_roc_auc = link_prediction(
        G_ppi, dim, p, q, l, l_ns, n_batches, batch_size, device)
    print(f'Dataset: PPI, Accuracy: {ppi_accuracy}, ROC-AUC: {ppi_roc_auc}')
