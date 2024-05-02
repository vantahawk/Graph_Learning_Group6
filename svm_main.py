#import sklearn as skl
import numpy as np
import networkx as nx
import pickle
from sklearn.svm import SVC, LinearSVC
#from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score#, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
#from threading import active_count
from psutil import cpu_count, virtual_memory


###

# import kernels here (add/choose/replace as needed):
# kernel functions need to return a flat array/list for each graph!

# David:
from Sheet1.David.exercise1.algeb_closed_walk import algeb_closed_walk_kernel # David's closed walk

# Ahmet:
#from Sheet1.Ahmet.exercise1.main import closed_walk_kernel # Ahmet's closed walk
from Sheet1.Ahmet.exercise2.main import sample_and_build_graphlets # Ahmet's graphlet
import Sheet1.Ahmet.exercise3.main as ahmetWL # Ahmet's WL

# Benedict:


# add/choose names of kernel functions accordingly:
kernels = {'Closed Walk': algeb_closed_walk_kernel, #closed_walk_kernel
           'Graphlet': sample_and_build_graphlets,
           'WL\t': ahmetWL.weisfeiler_lehman_graph_kernel
          }

#Note: svm_main does not execute Ahmet's graphlet & WL kernel properly, likely due to issue with svm_main
#output table still shows if only lines for David's or Ahmet's closed walk kernel are chosen
###


datasets = ['DD', 
            'ENZYMES',
            'NCI1']
k = 10 # k-fold cross validation

#SVM-modules:
classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=1000, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False)
#cache_size=200, 1000, max(np.floor(virtual_memory()[1]/1e6)-1000, 1000)
#kernel=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

#classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=False, verbose=0, random_state=None, max_iter=1000)
#fit_intercept=True

#classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
#not tested yet

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
        scores = cross_val_score(classifier, graph_vectors, graph_labels, scoring='accuracy', cv=k, n_jobs=-1, verbose=0, pre_dispatch='2*n_jobs', error_score='raise')
        #pre_dispatch='2*n_jobs', 'n_jobs', None, cpu_count()-1, active_count()-1
        
        mean = np.mean(scores)
        standard_deviation = np.mean(np.power(scores, 2)) - mean ** 2
        
        #print(str(mean*100) + "\u00b1" + str(standard_deviation*100) + "\t", end='')
        print(str(round(mean*100, 2)) + "\u00b1" + str(round(standard_deviation*100, 2)) + "\t", end='') # simple rounding
        #print(str(int(np.floor(mean * 100))) + "\u00b1" + str(int(np.ceil(standard_deviation * 100))) + "\t", end='') # rounded to full %; down for mean, up for std
        #print("test\u00b1test\t", end='')
        
print("\n")