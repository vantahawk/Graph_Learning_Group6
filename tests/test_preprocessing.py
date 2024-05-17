import pytest
import networkx as nx
import numpy as np
from typing import *
from torch import Tensor

class TestPreprocessing:


    def test_normalized_adjacency_matrix_single_graph(self):
        from src.preprocessing import normalized_adjacency_matrix

        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3, 4, 5])

        edges:List[Tuple[int, int]] = [tuple(edge) for edge in np.random.choice([1,2,3,4,5], (10, 2)).tolist()]
        graph.add_edges_from(edges)

        result:Tensor = normalized_adjacency_matrix(graph)
        result = result.to_dense().numpy()
        expected:np.ndarray = np.array([nx.adjacency_matrix(graph).toarray()])

        #test the shapes and whether the entries are still in the correct places
        assert result.shape == expected.shape
        assert np.all((result==0)==(expected==0))



