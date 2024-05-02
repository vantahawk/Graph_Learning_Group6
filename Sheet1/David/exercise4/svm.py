#import sklearn as skl
import numpy as np
import networkx as nx
import pickle
from sklearn.svm import SVC#, LinearSVC, SGDClassifier
from sklearn.model_selection import cross_val_score#, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from threading import active_count
from psutil import cpu_count, virtual_memory

# import kernels...


### Examp.: algeb_closed_walk_kernel

from networkx.linalg import adjacency_matrix
from scipy.linalg import eigvalsh
#from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import eigsh

def algeb_closed_walk_kernel(graph: nx.Graph, max_length: int = 20) -> np.ndarray:
    # eigenvalues of adjacency matrix; choose:
    # sparse method:
    #eigenvalues = eigsh(csr_matrix.asfptype(adjacency_matrix(graph)), k=nx.number_of_nodes(graph)-1, which='LM', maxiter=None, tol=0, return_eigenvectors=False, mode='normal')
    # or dense method:
    eigenvalues = eigvalsh(adjacency_matrix(graph).todense(), overwrite_a=False, check_finite=True, subset_by_index=None, driver=None) 
    # seems to use CPU-cores more efficiently than sparse method, also slightly more accurate (based on mean deviation w.r.t. taking l-powers of A directly)
    
    eigenvalues_power = eigenvalues
    kernel_vector = []
        
    for l in range(max_length):
        eigenvalues_power = np.multiply(eigenvalues_power, eigenvalues) # successively multiply eigenvalues elem.wise
        kernel_vector.append(np.sum(eigenvalues_power))
    
    return np.array(kernel_vector) # returns kernel vector for walk lengths l=2,...,max_length+1

###


# add names of kernel functions/methods:
kernels = {'Closed Walk': algeb_closed_walk_kernel
           #, 'Graphlet': #
           #, 'WL\t': #
          }
datasets = ['DD', 
            'ENZYMES',
            'NCI1']
k = 10 # k-fold cross validation

classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=1000, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
#cache_size=200, 1000, max(np.floor(virtual_memory()[1]/1e6)-1000, 1000)
#kernel=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
classifier = make_pipeline(StandardScaler(), classifier) # use std-normalization of kernel vector data

print("Mean \u00b1 Standard Deviation of Accuracy scores (rounded in %) for " + str(k) + "-fold Cross-validation:\n\nKernel \u2193 | Dataset \u2192\t", end='')
for dataset in datasets:
    print(dataset + "\t\t", end='')
print("\n", end='')

for kernel in kernels:
    print("\n" + kernel + "\t\t", end='')
    
    for dataset in datasets:
        with open('datasets/' + str(dataset) + '/data.pkl', 'rb') as data:
            graphs = pickle.load(data)
        graph_labels = [graph.graph['label'] for graph in graphs]
        graph_vectors = [kernels[kernel](graph) for graph in graphs] # kernel vectors of all graphs for given kernel & dataset
        
        #accuracy scores from k-fold cross validation:
        scores = cross_val_score(classifier, graph_vectors, graph_labels, scoring='accuracy', cv=k, n_jobs=-1, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan)
        #pre_dispatch='2*n_jobs', 'n_jobs', None, active_count()-1, cpu_count()-1
        
        mean = np.mean(scores)
        standard_deviation = np.mean(np.power(scores, 2)) - mean ** 2
        
        #print(str(mean*100) + "\u00b1" + str(standard_deviation*100) + "\t", end='')
        print(str(round(mean*100, 2)) + "\u00b1" + str(round(standard_deviation*100, 2)) + "\t", end='') # simple rounding
        #print(str(int(np.floor(mean * 100))) + "\u00b1" + str(int(np.ceil(standard_deviation * 100))) + "\t", end='') # rounded to full %; down for mean, up for std
        #print("test\u00b1test\t", end='')