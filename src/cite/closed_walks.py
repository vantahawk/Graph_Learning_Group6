# external imports:
#import numpy as np
from numpy import abs, dot, empty, isreal, max, min
#from scipy.linalg import eigh#, eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh#, eigs
from torch import Tensor, tensor

# internal imports:
from sparse_graph import Sparse_Graph



def CW_kernel(G: Sparse_Graph, l: int = 8,
              # number of (largest magnitude) EValue/EVector-pairs to be computed, n_EVal in [1,...,n_nodes-1], takes quite long for ca. >=1000:
              n_EVal: int = 500 #100 #200 #500 #1000  #G.n_nodes - 1
              ) -> Tensor:  # output (dtype=float64) overflows for l > 8...
    '''returns *node-lvl* closed walk kernel (CW) embedding matrix for given graph G (as sparse graph rep.), yields for each node its *approx.* number of closed walks of lengths [2,...,l]'''
    dim_cw = l - 1  # embedding dimension

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

    print("Compute CW embedding") if print_progress else print(end="")
    #return tensor(dot((EVec * EVec.conj()), EVal_powers))  # CW embedding matrix W, see [pdf-file] for derivation  # TODO add pdf-file(name)
    return tensor(dot(EVec ** 2, EVal_powers))  # in case EVec is real-valued (true for CITE)



print_progress = False
if __name__ == "__main__":
    # test CW embedding:
    import pickle
    print_progress = True

    #with open('datasets/Citeseer/data.pkl', 'rb') as data:
    #with open('datasets/Cora/data.pkl', 'rb') as data:
    #with open('datasets/Facebook/data.pkl', 'rb') as data:  # cannot construct self.node_labels for Facebook, idk why, not needed tho
    #with open('datasets/PPI/data.pkl', 'rb') as data:
    with open('datasets/CITE/data.pkl', 'rb') as data:
    #with open('datasets/LINK/data.pkl', 'rb') as data:
        graph = pickle.load(data)#[0]

    W = CW_kernel(Sparse_Graph(graph, False))
    print(f"\n{W[: 5]}\nshape: {W.shape}")
