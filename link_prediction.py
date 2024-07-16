import pickle
import networkx as nx
import numpy as np


class TransE:
    """
    TransE model for learning low-dimensional embeddings of entities and relationships
    in a knowledge graph, and for predicting missing relationships.

    Attributes:
        n_entities (int): Number of unique entities in the graph.
        n_relations (int): Number of unique relationships in the graph.
        dim (int): Dimensionality of the embeddings.
        lr (float): Learning rate for the stochastic gradient descent.
        margin (float): Margin hyperparameter for the margin-based ranking loss.
        entity_embeddings (np.ndarray): Embeddings for the entities.
        relation_embeddings (np.ndarray): Embeddings for the relationships.
    """

    def __init__(self, n_entities, n_relations, dim, lr, margin):
        """
        Initialize the TransE model with random embeddings for entities and relationships.

        Args:
            n_entities (int): Number of unique entities in the graph.
            n_relations (int): Number of unique relationships in the graph.
            dim (int): Dimensionality of the embeddings.
            lr (float): Learning rate for the stochastic gradient descent.
            margin (float): Margin hyperparameter for the margin-based ranking loss.
        """
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.lr = lr
        self.margin = margin

        # Initialize entity and relation embeddings with small random values
        self.entity_embeddings = np.random.uniform(
            -6/np.sqrt(dim), 6/np.sqrt(dim), (n_entities, dim))
        self.relation_embeddings = np.random.uniform(
            -6/np.sqrt(dim), 6/np.sqrt(dim), (n_relations, dim))

        # Normalize the entity embeddings to unit length
        self.entity_embeddings = self.entity_embeddings / \
            np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)

    def train_step(self, h, t, r, h_neg, t_neg):
        """
        Perform a single training step using a mini-batch of positive and negative samples.

        Args:
            h (np.ndarray): Indices of the head entities in the positive samples.
            t (np.ndarray): Indices of the tail entities in the positive samples.
            r (np.ndarray): Indices of the relationships in the positive samples.
            h_neg (np.ndarray): Indices of the head entities in the negative samples.
            t_neg (np.ndarray): Indices of the tail entities in the negative samples.

        Returns:
            float: The loss value for the training step.
        """
        # Get embeddings for positive and negative samples
        h_e = self.entity_embeddings[h]
        t_e = self.entity_embeddings[t]
        r_e = self.relation_embeddings[r]
        h_neg_e = self.entity_embeddings[h_neg]
        t_neg_e = self.entity_embeddings[t_neg]

        # Compute distances for positive and negative samples
        pos_dist = np.linalg.norm(h_e + r_e - t_e, axis=1)
        neg_dist = np.linalg.norm(h_neg_e + r_e - t_neg_e, axis=1)

        # Calculate the margin-based loss
        loss = np.maximum(0, self.margin + pos_dist - neg_dist).sum()

        # Compute gradients for the embeddings
        grad_pos = 2 * (h_e + r_e - t_e)
        grad_neg = 2 * (h_neg_e + r_e - t_neg_e)

        # Update embeddings based on the gradients
        for i in range(len(h)):
            if pos_dist[i] + self.margin > neg_dist[i]:
                self.entity_embeddings[h[i]] -= self.lr * grad_pos[i]
                self.entity_embeddings[t[i]] += self.lr * grad_pos[i]
                self.relation_embeddings[r[i]] -= self.lr * grad_pos[i]
                self.entity_embeddings[h_neg[i]] += self.lr * grad_neg[i]
                self.entity_embeddings[t_neg[i]] -= self.lr * grad_neg[i]

        # Re-normalize the entity embeddings to unit length after update
        self.entity_embeddings = self.entity_embeddings / \
            np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        return loss

    def train(self, train_data, n_epochs):
        """
        Train the TransE model using the provided training data.

        Args:
            train_data (list of tuples): List of training samples, each represented as a tuple (h, t, r).
            n_epochs (int): Number of epochs to train the model.
        """
        # Train the model for a specified number of epochs
        for epoch in range(n_epochs):
            # Shuffle the training data
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), 128):
                # Process mini-batches of the training data
                batch = train_data[i:i+128]
                h, t, r = zip(*batch)
                h = np.array(h)
                t = np.array(t)
                r = np.array(r)

                # Generate negative samples by randomly corrupting head or tail entities
                h_neg = np.random.randint(0, self.n_entities, len(h))
                t_neg = np.random.randint(0, self.n_entities, len(t))

                # Perform a training step
                self.train_step(h, t, r, h_neg, t_neg)

    def predict(self, h, t):
        """
        Predict the relationship labels for given head and tail entities.

        Args:
            h (np.ndarray): Indices of the head entities.
            t (np.ndarray): Indices of the tail entities.

        Returns:
            np.ndarray: Predicted relationship labels.
        """
        # Get embeddings for head and tail entities
        h_e = self.entity_embeddings[h]
        t_e = self.entity_embeddings[t]

        # Compute scores for all possible relations
        scores = np.linalg.norm(
            h_e[:, np.newaxis, :] + self.relation_embeddings - t_e[:, np.newaxis, :], axis=2)

        # Return the relation with the minimum score (closest in embedding space)
        return np.argmin(scores, axis=1)


if __name__ == "__main__":

    # Load the graph data
    with open("datasets/LINK/data.pkl", "rb") as f:
        data = pickle.load(f)
        G = nx.MultiDiGraph(data)

    # Separate the edges into training and test sets
    train_edges = []
    test_edges = []

    # Iterate over all edges in the graph
    for u, v, data in G.edges(data=True):
        if data['edge_label'] is None:
            test_edges.append((u, v, data['id']))
        else:
            train_edges.append((u, v, data['edge_label']))

    # Extract unique nodes and relationships
    nodes = list(G.nodes())
    relations = list(set(edge[2] for edge in train_edges))

    # Create dictionaries to map nodes and relationships to unique indices
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    rel2idx = {rel: idx for idx, rel in enumerate(relations)}
    idx2rel = {idx: rel for rel, idx in rel2idx.items()}

    # Convert edges to indices for easier processing in the TransE model
    train_edges = [(node2idx[u], node2idx[v], rel2idx[rel])
                   for u, v, rel in train_edges]
    test_edges = [(node2idx[u], node2idx[v], edge_id)
                  for u, v, edge_id in test_edges]

    # Define parameters for the TransE model
    dim = 50
    lr = 0.01
    margin = 1.0
    n_epochs = 100

    # Initialize and train the TransE model
    model = TransE(len(nodes), len(relations), dim, lr, margin)
    model.train(train_edges, n_epochs)

    # Predict missing labels for test edges
    test_h, test_t, edge_ids = zip(*test_edges)
    predicted_labels = model.predict(np.array(test_h), np.array(test_t))

    # Pair edge IDs with their corresponding predictions
    id_prediction_pairs = list(zip(edge_ids, predicted_labels))

    # Sort predictions by edge IDs
    id_prediction_pairs.sort(key=lambda x: x[0])

    # Extract sorted predictions
    sorted_predictions = [pred for _, pred in id_prediction_pairs]

    # Ensure the predictions are in the required format
    pred = [int(pred) for pred in sorted_predictions]

    # Save predictions to file
    with open("LINK-Predictions.pkl", "wb") as f:
        pickle.dump(pred, f)
