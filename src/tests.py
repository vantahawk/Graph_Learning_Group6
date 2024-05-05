import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from kernels.base import new_color_hash
from kernels.graphlet import GraphletKernel as GK

def test_new_color_hash():
    initial_colors = np.array([[1,1,1,1,1]])
    g = nx.Graph()
    g.add_nodes_from(range(5))
    g.add_edges_from([(0,1), (1,2), (2,3), (3,4), (3,1), (2,4)])

    new_colors = new_color_hash(initial_colors, [g])
    print(new_colors)
    #add colors as labels
    for i, node in enumerate(g.nodes):
        g.nodes[node]["color"] = new_colors[0][i]
    nx.draw(g, with_labels=True, labels=nx.get_node_attributes(g, "color"))
    plt.show()

    new_colors = new_color_hash(new_colors, [g])
    for i, node in enumerate(g.nodes):
        g.nodes[node]["color"] = new_colors[0][i]
    nx.draw(g, with_labels=True, labels=nx.get_node_attributes(g, "color"))
    plt.show()

def test_iso_graph_creation():
    GK.compute_iso_graphs(6, True)


if __name__ == "__main__":
    test_iso_graph_creation()