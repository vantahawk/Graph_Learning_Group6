import networkx as nx
from collections import defaultdict, Counter
from typing import List


def initial_coloring(graph: nx.Graph, default_color=0):
    """Initialize coloring of the graph nodes. Use node labels if available, otherwise use default color."""
    return {node: graph.nodes[node].get('label', default_color) for node in graph.nodes}


def refine_colors(graph: nx.Graph, current_colors, color_map):
    """Refine colors based on current colors and neighborhood colors, using a color map to ensure unique integers."""
    new_colors = {}
    next_color = max(color_map.values(), default=0) + 1  # Get next color code
    for node in graph.nodes:
        neighbors = graph.neighbors(node)
        # Sort the colors to ensure consistency
        neighbor_colors = tuple(
            sorted(current_colors[neighbor] for neighbor in neighbors))
        # Create a unique key for the current color and its neighborhood
        color_key = (current_colors[node], neighbor_colors)
        if color_key not in color_map:
            color_map[color_key] = next_color
            next_color += 1
        new_colors[node] = color_map[color_key]
    return new_colors


def compute_histogram(colors: dict):
    """Compute the histogram of the colors."""
    color_count = Counter(colors.values())
    return color_count


def weisfeiler_lehman_graph_kernel(graphs: List[nx.Graph], num_rounds=4):
    """Compute the Weisfeiler-Lehman kernel for a list of graphs."""
    # Initialize coloring for all graphs
    all_colors = [initial_coloring(graph) for graph in graphs]
    # Initialize color map for consistent coloring across graphs
    color_map = {}

    # Iteratively refine colors
    for _ in range(num_rounds):
        all_colors = [refine_colors(graph, colors, color_map)
                      for graph, colors in zip(graphs, all_colors)]

    # Compute histograms
    histograms = [compute_histogram(colors) for colors in all_colors]

    # Normalize histograms by ensuring each histogram has the same features
    all_features = set().union(*[hist.keys() for hist in histograms])
    normalized_histograms = []
    for hist in histograms:
        normalized_hist = {feature: hist.get(
            feature, 0) for feature in all_features}
        normalized_histograms.append(normalized_hist)

    return normalized_histograms


# Example usage with NetworkX graphs
G1 = nx.Graph()
G1.add_nodes_from([1, 2, 3], label=1)
G1.add_edges_from([(1, 2), (2, 3)])

G2 = nx.Graph()
G2.add_nodes_from([1, 2, 3, 4], label=2)
G2.add_edges_from([(1, 2), (1, 3), (1, 4)])

graphs = [G1, G2]
kernel = weisfeiler_lehman_graph_kernel(graphs)
print(kernel)
