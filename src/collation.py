import torch as th


def custom_collate(sparse_rep_list: list[tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    '''custom collation function, takes list of sparse representations of graphs in batch it, returns custom sparse representation of collated batch graph'''
    graph_index, idx_shift = 0, 0  # initialize iterators
    edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = [
    ], [], [], [], []  # collect each component of sparse rep. of batch in resp. list

    # run over sparse rep. of each graph in batch
    for edge_idx, node_features, edge_features, graph_label in sparse_rep_list:
        n_nodes = len(node_features)  # number of nodes in current graph
        # append each component list
        node_features_col.append(node_features)
        edge_features_col.append(edge_features)
        graph_labels_col.append(graph_label)
        # append edge_idx of current graph w/ elements shifted by number of nodes in all graphs up to this point (idx_shift)
        edge_idx_col.append(edge_idx + idx_shift * th.ones(edge_idx.shape))
        # append batch_idx by repeating graph_index for each node in current graph
        batch_idx.append(graph_index * th.ones(n_nodes))
        # update iterators
        idx_shift += n_nodes
        graph_index += 1

    # return concatenated sparse representation for collated graph of batch
    return th.cat(edge_idx_col, -1).type(th.long), th.cat(node_features_col, 0), th.cat(edge_features_col, 0), th.cat(graph_labels_col, 0), th.cat(batch_idx, 0).type(th.long)
