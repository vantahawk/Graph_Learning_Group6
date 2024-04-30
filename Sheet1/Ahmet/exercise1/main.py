import numpy as np
import networkx as nx


def closed_walk_kernel(graph: nx.Graph, max_length: int) -> np.ndarray:
    # Get the adjacency matrix from the graph
    A = nx.adjacency_matrix(graph).todense()
    # Convert to numpy array for power operations
    A = np.array(A, dtype=np.float64)

    # Initialize the feature vector
    feature_vector = np.zeros(max_length)

    # Current power of adjacency matrix (start with the first power, which is the matrix itself)
    A_power = A.copy()

    # Compute closed walks of each length from 1 to max_length
    for length in range(1, max_length + 1):
        if length > 1:
            # Compute the next power of A
            A_power = np.linalg.matrix_power(A, length)

        # The trace of A_power gives us the number of closed walks of current length
        feature_vector[length - 1] = np.trace(A_power)

    return feature_vector


# Example usage with a simple graph
G = nx.Graph()
# Adding edges to the graph
edges = [(1, 2), (2, 3), (3, 1), (3, 4)]  # A triangle with a tail
G.add_edges_from(edges)

# Calculate closed walk kernel feature vector for walks up to length 5
feature_vector = closed_walk_kernel(G, 5)
print("Feature vector of closed walks:", feature_vector)
