import torch as th



def custom_collate(sparse_rep_list: list[tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    '''custom collation function, takes list of sparse representations of graphs in batch it, returns custom sparse representation of collated batch graph
    
    Args: 
    - sparse_rep_list: The sparse representation list.
        A list over all the graphs, that shall be batched together.
        The inner tuple holds the sparse representation of a single graph: the edge indexes, the node features, the edge features, and the graph label.

    Returns:
    - edge_idx_col: The edge indexes of the collated batch graph.
    - node_features_col: The node features of the collated batch graph.
    - edge_features_col: The edge features of the collated batch graph.
    - graph_labels_col: The graph labels of the collated batch graph.
    - batch_idx: The batch indexes of the collated batch graph. Via this the original nodes are identifiable.
    '''
    
    edge_idx_col, node_features_col, edge_features_col, graph_labels_col, batch_idx = [], [], [], [], []  # collect each component of sparse rep. of batch in resp. list
    device = sparse_rep_list[0][0].device  # get device of input tensors
    #do the for loop in parallel
    graph_index, id_shift = 0, 0  # initialize iterators
    graph_index = th.tensor(graph_index, device=device)
    id_shift = th.tensor(id_shift, device=device)
    for edge_idx, node_features, edge_features, graph_label in sparse_rep_list:  # run over sparse rep. of each graph in batch
        n_nodes = len(node_features)  # number of nodes in current graph
        # append each component list
        node_features_col.append(node_features)
        edge_features_col.append(edge_features)
        #graph_labels_col.append(graph_label)
        graph_labels_col.append([graph_label])
        # append edge_idx of current graph w/ elements shifted by number of nodes in all graphs up to this point (idx_shift)
        edge_idx_col.append(edge_idx + id_shift)
        batch_idx.append(graph_index * th.ones(n_nodes, device=device))  # append batch_idx by repeating graph_index for each node in current graph
        # update iterators
        id_shift += n_nodes
        graph_index += 1

    # return concatenated sparse representation for collated graph of batch
    return (
        th.cat(edge_idx_col, -1).type(th.long), 
        th.cat(node_features_col, 0).type(th.float), 
        th.cat(edge_features_col, 0).type(th.float), 
        th.tensor(graph_labels_col, device=device).type(th.float), 
        th.cat(batch_idx, 0).type(th.long)
    )
    #th.cat(graph_labels_col, 0)
