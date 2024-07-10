# external imports:
#import numpy as np
from numpy import abs, dot, empty, isreal, max, min, float64
#from scipy.linalg import eigh#, eig
from scipy.sparse import csr_array, csr_matrix, eye_array
from scipy.sparse.linalg import eigsh#, eigs
#import torch as th
from torch import Tensor, tensor
from torch import float64 as th_float64

# internal imports:
from sparse_graph import Sparse_Graph



def CW_kernel(G: Sparse_Graph, l: int = 8,
              # number of (largest magnitude) EValue/EVector-pairs to be computed, n_EVal in [1,...,n_nodes-1], takes quite long for ca. >=1000:
              #n_EVal: int = 500 #100 #200 #500 #1000  #G.n_nodes - 1
              ) -> Tensor:  # output (dtype=float64) overflows for l > 8...
    '''returns *node-lvl* closed walk kernel (CW) embedding matrix for given graph G (as sparse graph rep.), yields for each node its *approx.* number of closed walks of lengths [2,...,l]'''
    dim_cw = l - 1  # embedding dimension
    """#
    ## spectral method:
    # compute eigenvalue vector EVal (real-valued) & eigenvector matrix EVac (unitary) of adjacency matrix of (undirected) graph G:
    print("Solve eigenvalue problem") if print_progress else print(end="")
    EVal, EVec = eigsh(csr_matrix.asfptype(G.adj_mat), k=n_EVal, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal')  #G.adj_mat.todense()  #G.adj_mat.toarray()  #csr_matrix.asfptype(G.adj_mat)

    if print_progress:
        EVal_abs = abs(EVal)  # absolute values of EVal
        print(f"EVal_min = {min(EVal_abs, axis=0)}, EVal_max = {max(EVal_abs, axis=0)}")  # smallest & largest, computed eigenvalue by magnitude
        print(f"EVec is real?: {isreal(EVec).all()}")  # all eigenvectors are real-valued for CITE

    # compute eigenvalue vector EVal (complex-valued) & eigenvector matrix EVac (unitary) of adjacency matrix of (directed) graph G:
    #EVal, EVec = eigs(G.adj_mat, k=n_EVal, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)  #adj_mat.todense()  #G.adj_mat.toarray()

    print("Compute eigenvalue power matrix") if print_progress else print(end="")
    EVal_powers = empty((n_EVal, dim_cw))  # matrix of powers of eigenvalues, when filled up: EVal_powers[:, j] = EVal**(j+2)
    EVal_power = EVal
    for j in range(dim_cw):  # fill EVal_powers up
        EVal_power *= EVal  # take next power of EVal
        EVal_powers[:, j] = EVal_power  # assign to resp. column in EVal_powers

    #print(f"EVec.dtype = {EVec.dtype}, EVal_powers.dtype = {EVal_powers.dtype}")
    print("Compute CW embedding") if print_progress else print(end="")
    #W = dot((EVec * EVec.conj()), EVal_powers)  # CW embedding matrix W, see [pdf-file] for derivation  # TODO add pdf-file(name)
    W = dot(EVec ** 2, EVal_powers)  # in case EVec is real-valued (true for CITE)
    """#
    ##direct mat.mult. method:
    #adj_diag_powers = csr_array((G.n_nodes, dim_cw))  # csr-array w/ each column j as flattened diagonal of adj_mat**(j+2)
    print("Compute CW embedding directly") if print_progress else print(end="")
    adj_diag_powers = empty((G.n_nodes, dim_cw))  # np.array
    adj_mat = G.adj_mat.astype(float64)
    adj_mat_power = adj_mat
    for j in range(dim_cw):  # fill diag_powers up
        adj_mat_power @= adj_mat  # take next power of adj_mat
        adj_diag_powers[:, j] = adj_mat_power.diagonal()  # assign its diagonal array to resp. column in diag_powers
    W = adj_diag_powers
    #
    #return tensor(dot(EVec ** 2, EVal_powers)).type(th_float64)  # in case EVec is real-valued (true for CITE)
    return tensor(W).type(th_float64)
    #return W



print_progress = False
if __name__ == "__main__":
    # test CW embedding:
    import pickle
    from timeit import default_timer
    print_progress = True

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    l = 21
    t_start = default_timer()
    W = CW_kernel(Sparse_Graph(graph, False), l)
    print(f"l = {l}, Time = {(default_timer() - t_start) / 60} mins\n{W[: 5]}\nshape = {W.shape}")  #, dtype = {W.dtype}
