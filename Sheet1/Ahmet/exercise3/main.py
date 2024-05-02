import networkx as nx
from collections import defaultdict, Counter
from typing import List
import numpy as np


def initial_coloring(graph: nx.Graph, default_color=0):
    """Colors the graph nodes by using the provided node labels if available and a default color as fallback if not."""
    return {node: graph.nodes[node].get('label', default_color) for node in graph.nodes}


def refine_colors(graph: nx.Graph, current_colors, color_map):
    """Basically the function that is called in the loop. It refines the colors of the nodes based on the current colors and the neighboring nodes' colors."""
    new_colors = {}  # of form {node: color}
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
        # Assign the new color to the node
        new_colors[node] = color_map[color_key]
    return new_colors


def weisfeiler_lehman_graph_kernel(graph: nx.Graph, num_rounds=4):
    """Compute the Weisfeiler-Lehman kernel for a single graph."""
    # Initialize coloring for the graph
    colors = initial_coloring(graph)
    # Initialize color map for consistent coloring across graphs
    color_map = {}

    # Iteratively refine colors
    for _ in range(num_rounds):
        colors = refine_colors(graph, colors, color_map)

    # Compute histogram
    histogram = Counter(colors.values())

    # Get the maximum color value and initialize a 1D numpy array with zeros
    max_color = max(histogram.keys())
    color_histogram = np.zeros(max_color + 1)

    # Fill the histogram values in the 1D numpy array with colors as indices
    for color, count in histogram.items():
        color_histogram[color] = count

    return color_histogram
