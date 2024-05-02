import numpy as np
import networkx as nx
from typing import Tuple, List


def sample_and_build_graphlets(graph: nx.Graph, num_samples: int = 1000) -> Tuple[List[nx.Graph], np.ndarray]:
    nodes = np.array(graph.nodes())
    all_graphlets: List[nx.Graph] = []
    # Preallocate with expected size if known, else start small and dynamically resize if necessary.
    # 34 is the maximum number of unique graphlets for 5 nodes.
    # But this won't be a problem since the graphlet_counts array will be resized if needed.
    graphlet_counts = np.zeros(34, dtype=int)

    for i in range(num_samples):
        sampled_nodes = np.random.choice(nodes, 5, replace=False)
        # Make a copy of the subgraph
        subgraph = nx.Graph(graph.subgraph(sampled_nodes))
        subgraph = nx.convert_node_labels_to_integers(
            subgraph)  # Standardize node labels

        # Check for isomorphism with known graphlets
        found_isomorph = False
        for idx, g in enumerate(all_graphlets):
            if nx.is_isomorphic(subgraph, g):
                graphlet_counts[idx] += 1
                found_isomorph = True
                break

        # If no isomorph found, add the new graphlet to the list
        if not found_isomorph:
            all_graphlets.append(subgraph)
            index = len(all_graphlets) - 1
            if index >= len(graphlet_counts):  # Ensure there is room in the array
                graphlet_counts = np.pad(
                    graphlet_counts, (0, 10), mode='constant')  # Increase size by 10
            graphlet_counts[index] = 1
            print(f"Sample {i}: New graphlet added as graphlet {index + 1}")

    # Trim graphlet_counts to match number of unique graphlets
    return all_graphlets, graphlet_counts[:len(all_graphlets)]


# Example usage:
# Create a larger example graph
G = nx.erdos_renyi_graph(10, 0.5)

# Get the feature vector for the graph using the Graphlet Kernel
graphlets, feature_vector = sample_and_build_graphlets(G)
print("Number of distinct graphlets found:", len(graphlets))
print("Feature vector:", feature_vector)
