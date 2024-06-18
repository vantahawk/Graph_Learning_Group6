import torch
import networkx as nx
import random
from torch.utils.data import IterableDataset


class RandomWalkDataset(IterableDataset):
    """
    An iterable dataset for generating random walks from a graph. This dataset is specifically tailored for node2vec
    embeddings, where the random walks are influenced by parameters p, q, and are of a fixed length l. Additionally,
    it also samples negative nodes for contrastive learning purposes.

    Attributes:
        G (networkx.Graph): The graph from which to generate random walks.
        p (float): Return hyperparameter, which controls the likelihood of immediately revisiting a node in the walk.
        q (float): In-out parameter, which controls the search to differentiate between inward and outward nodes.
        l (int): Length of each random walk.
        l_ns (int): Number of negative samples to generate per walk.

    Methods:
        __iter__():
            Provides an iterator that yields batches of start nodes, corresponding walks, and negative samples.

        random_walk_generator():
            A generator that produces unlimited random walks and negative samples.

        choose_next_node(current_node, previous_node):
            Chooses the next node in a random walk based on the node2vec sampling strategy.

        sample_negative_nodes(walk, num_samples):
            Samples nodes not present in the current walk, ensuring each sample is unique.
    """

    def __init__(self, G: nx.Graph, p: float, q: float, l: int, l_ns: int):
        """
        Initializes the dataset with a given graph and random walk parameters.

        Args:
            G (networkx.Graph): The graph from which random walks will be generated.
            p (float): Return hyperparameter.
            q (float): In-out parameter.
            l (int): The fixed length of the random walks.
            l_ns (int): Number of negative samples to generate per walk.
        """
        super().__init__()
        self.G = G
        self.p = p
        self.q = q
        self.l = l
        self.l_ns = l_ns

    def __iter__(self):
        """
        Provides an iterator that yields batches of start nodes, corresponding walks, and negative samples.
        Each batch contains one example of these components.

        Yields:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors of start node, walk nodes, and negative samples.
        """
        return self.random_walk_generator()

    def random_walk_generator(self):
        """
        Generator that yields unlimited random walks from the graph G, using parameters p, q, and l.

        Yields:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Each yield returns tensors for start node,
            nodes in the walk, and negative sample nodes.
        """
        while True:
            start_node = random.choice(list(self.G.nodes))
            walk = [start_node]
            current_node = start_node

            for i in range(1, self.l):
                v_neighbors = list(self.G.neighbors(current_node))
                if not v_neighbors:
                    break

                if i == 1:
                    current_node = random.choice(v_neighbors)
                else:
                    current_node = self.choose_next_node(
                        current_node, walk[-2])

                walk.append(current_node)

            negative_samples = self.sample_negative_nodes(set(walk), self.l_ns)
            yield torch.tensor(start_node), torch.tensor(walk), torch.tensor(negative_samples)

    def choose_next_node(self, current_node: int, previous_node: int) -> int:
        """
        Chooses the next node in the walk using node2vec sampling strategy based on parameters p and q.

        Args:
            current_node (int): The current node from which the next node is to be chosen.
            previous_node (int): The previous node in the walk.

        Returns:
            int: The chosen next node for the walk.
        """
        neighbors = list(self.G.neighbors(current_node))
        if not neighbors:
            return current_node

        weights = [1 / self.p if neighbor == previous_node else 1 /
                   self.q if neighbor not in self.G.neighbors(previous_node) else 1 for neighbor in neighbors]
        probabilities = [w / sum(weights) for w in weights]
        return random.choices(neighbors, weights=probabilities, k=1)[0]

    def sample_negative_nodes(self, walk: set, num_samples: int) -> list:
        """
        Samples negative nodes that are not present in the current walk.

        Args:
            walk (set): A set of nodes that are part of the current walk.
            num_samples (int): The number of negative samples to generate.

        Returns:
            list: A list of nodes that are not part of the current walk.
        """
        negative_samples = []
        walk_nodes = set(walk)
        all_nodes = list(self.G.nodes)

        i = 0
        while i < num_samples:
            node = random.choice(all_nodes)

            if node not in walk_nodes:
                negative_samples.append(node)
                walk_nodes.add(node)
                i += 1

        return negative_samples
