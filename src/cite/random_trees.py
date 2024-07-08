'''implementation of iterable dataset for streaming random trees'''
# external imports:
#from networkx import Graph, adjacency_matrix, number_of_edges, number_of_nodes
import numpy as np  # TODO replace np. w/ direct imports (maybe)
from numpy.random import default_rng
from scipy.sparse import eye_array
from torch.utils.data import IterableDataset#, get_worker_info
from typing import Iterator

# internal imports:
#from node2vec import Node2Vec, train_node2vec
#from random_walks import RW_Iterable
#from node_class import node_class
from sparse_graph import Sparse_Graph



class RT_Iterable(IterableDataset):  # TODO parallelize/find multi-process slu.
    '''implements custom iterable dataset for random [pm]-trees with m+1 nodes over given sparse graph rep. G, together w/ [m_ns] negative samples'''
    def __init__(self, G: Sparse_Graph, p: float, m: int, m_ns: int, batch_size: int) -> None:
        super().__init__()

        # main graph attributes:
        self.n_nodes = G.n_nodes
        #self.adj_mat = G.adj_mat.toarray()
        self.adj_mat = (G.adj_mat + eye_array(self.n_nodes, format='csr')).toarray()  # w/ added self-loops, allows for undirec.-graph-shortcut

        # attributes for random walk generation:
        self.rng = default_rng(seed=None)  # default RNG object for general random sampling
        self.p = p  # in ]0,1], relative share of nodes to be sampled from each new tree frontier (see below), low: sparse & broad tree, high: dense & concentrated tree
        self.m = m  # random tree size, i.e. number of nodes in tree excl. start node
        self.tree_dim = self.m + 1  # (incl. start node)
        self.m_ns = m_ns  # number of negative samples
        self.batch_size = batch_size  # size of random tree batch generated by rw_batch()


    def rt_batch(self) -> list[np.ndarray]:
        '''returns batch (list) of p-tree data, each including: random start node, followed by m nodes of random p-tree, followed by m_ns negative samples, concatenated into 1D-np.ndarray'''
        batch = np.empty((self.batch_size, self.tree_dim + self.m_ns), dtype=np.int64)  # ...in batch np.ndarray  #dtype=np.int32
        #batch = []  # collect trees in batch list

        tree = 0  # counts (index of) trees in current batch
        #for tree in range(self.batch_size):
        while tree < self.batch_size:  # while tree batch not full yet
            print(f"tree {tree}: ", end="") if print_progress else print(end="")
            start = self.rng.choice(self.n_nodes, size=1, replace=False, p=None, axis=0, shuffle=False)  # uniformly sampled start node, np.int64
            p_tree_list = [start]  # collect sampled nodes of p-tree (list)
            p_tree_idx = np.zeros(self.n_nodes)  # tree idx: analogous indexing array of p-tree
            p_tree_idx[start] = 1  # add start node to tree idx
            n_tree = 0  # number of nodes (excl. start node) in tree sor far

            # sample the m next nodes in p-tree using algebraic construction
            while n_tree < self.m:
                # frontier: indexing array of all potential next nodes in tree:
                #frontier = (1 - p_tree_idx) * np.max(self.adj_mat * np.reshape(p_tree_idx, (self.n_nodes, 1)), axis=0)  # general version
                #frontier = (1 - p_tree_idx) * np.max(self.adj_mat * p_tree_idx, axis=-1)  # shortcut for undirected graphs w/o added self-loops
                frontier = np.max(self.adj_mat * p_tree_idx, axis=-1) - p_tree_idx  # shortcut for undirected graphs w/ added self-loops
                frontier_size = np.sum(frontier)  # number of nodes in [frontier]
                # for rounding n_next_nodes up:
                frontier_too_small = (frontier_size == 0)  # not enough nodes in frontier to sample from
                if frontier_too_small:
                    print("cancel") if print_progress else print(end="")
                    break  # stop trying to sample from current frontier
                #
                # number of [next_nodes] in tree as an approx. share [p] of nodes from [frontier], or else remainder of nodes up to m:
                n_next_nodes = min(int(self.p * frontier_size) + 1, self.m - n_tree)  # rounds up
                #n_next_nodes = min(round(self.p * frontier_size), self.m - n_tree)  # rounds evenly
                """# for rounding n_next_nodes evenly:
                frontier_too_small = (n_next_nodes == 0)  # not enough nodes in frontier to sample from
                if frontier_too_small:
                    print("cancel") if print_progress else print(end="")
                    break  # stop trying to sample from current frontier
                """#
                # uniformly sample [next_nodes] in tree, w/o repetition:
                next_nodes = self.rng.choice(self.n_nodes, size=n_next_nodes, replace=False, p = frontier / frontier_size, axis=0, shuffle=False)
                p_tree_list.append(next_nodes)  # add [next_nodes] to tree
                p_tree_idx[next_nodes] = 1  # update tree idx accordingly
                n_tree += n_next_nodes  # update n_tree
                print(f"{n_tree}, ", end="") if print_progress else print(end="")

            if frontier_too_small:
                continue  # skip current p-tree array, cancel/discard sampling for it, restart w/ new random start node
            p_tree = np.concatenate(p_tree_list, axis=-1)  # concatenate list of next nodes into array of all tree nodes
            rest_nodes = np.setdiff1d(np.arange(self.n_nodes), p_tree, assume_unique=True)  # remaining nodes after drawn p-tree
            # [m_ns] negative samples (np.ndarray) uniformly drawn from remaining nodes w/o repetition:
            neg_samples = self.rng.choice(rest_nodes, size=self.m_ns, replace=False, p=None, axis=0, shuffle=False)
            batch[tree] = np.concatenate([p_tree, neg_samples], axis=-1)  # concatenate tree & negative samples into p-tree data vector, add to batch
            #batch.append(np.concatenate([p_tree, neg_samples], axis=-1))
            tree += 1  # update tree index
            print("done") if print_progress else print(end="")

        print("\n") if print_progress else print(end="")
        return list(batch)
        #return batch


    def __iter__(self) -> Iterator[np.ndarray]:
        '''returns iterator of p-tree data, single-process: runs slow, unreliable or unintended otherwise...'''
        return iter(self.rt_batch())



print_progress = False
if __name__ == "__main__":
    # for testing streaming, batching, multiprocessing, etc.:
    import pickle
    from psutil import cpu_count
    from torch.utils.data import DataLoader, get_worker_info
    print_progress = True

    #device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    #n_workers = 2 #cpu_count(logical=True)
    batch_size =  5 #3 * n_workers
    p = 0.5 #0.1 #0.2 #0.5 #0.8 #1.0
    m = 10 #10 #20 #50 #100
    m_ns = 10 #10 #20 #50 #100
    set_node_labels = False

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    dataset = RT_Iterable(Sparse_Graph(graph, set_node_labels), p, m, m_ns, batch_size)  # custom iterable dataset instance
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # turns random walk batches into th.tensors, single-process

    # multi-process part removed
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, worker_init_fn=worker_split)  # issue w/ worker_split(), produces failed subprocesses
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)  # works but returns one batch for each subprocess? also duplicates depending on implementation (expected but not efficiently fixed yet)

    for run in range(3):
        for batch in dataloader:
            print(batch)
        print("\n")
