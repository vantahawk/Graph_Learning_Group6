from kernels import KernelName, KerneledGraph
from decorators import parseargs
import importlib
from typing import List, Dict

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score#, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler#, MinMaxScaler

import numpy as np
import pickle

kernelNameDict:Dict[str, str] = {
    "closed_walk": "ClosedWalkKernel",
    "graphlet": "GraphletKernel",
    "wl": "WLKernel"
}

datasets:List[str] = [
    'DD', 
    'ENZYMES',
    'NCI1'
]

defaultKernelArgs:Dict[str, Dict[str, str]] = {
    "closed_walk": {
        "max_length" : 20
    },
    "graphlet": {
        "k":5
        # "m": some value
    },
    "wl": {
        "k":5
    }
}

@parseargs(
    kernelname={
        "default":"closed_walk", 
        "type":str, 
        "help":"The name of the kernel to use, one of: \"closed_walk\", \"graphlet\", \"wl\"",
        "flags":["kn"]
    }, 
    cv={
        "default":10,
        "type":int,
        "help":"The number of folds for cross validation."
    },
    __description="The entry point of our graph kernel implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
     __help=True
)
def main(kernelname:KernelName, cv:int):
    print(f"Using kernel {kernelname}.")

    kernel = importlib.import_module(f"kernels.{kernelname}")
    
    #SVM-modules:
    classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=1000, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False)
    #cache_size=200, 1000, max(np.floor(virtual_memory()[1]/1e6)-1000, 1000)
    #kernel=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

    #classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=False, verbose=0, random_state=None, max_iter=1000)
    #fit_intercept=True

    #classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
    #not tested yet

    classifier = make_pipeline(StandardScaler(), classifier) # use std-normalization of kernel vector data

    
    for dataset in datasets:
        dataset_path:str = f'datasets/{dataset}/data.pkl'
        
        kerneledGraphs: List[KerneledGraph] = getattr(kernel, kernelNameDict[kernelname]).readGraphs(graphPath=dataset_path, **defaultKernelArgs[kernelname])

        with open(dataset_path, 'rb') as data:
            graphs = pickle.load(data)
            graph_labels = [graph.graph['label'] for graph in graphs]
            del graphs
            
            #accuracy scores from k-fold cross validation:
            scores = cross_val_score(classifier, kerneledGraphs, graph_labels, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0, pre_dispatch='2*n_jobs', error_score='raise')
            #pre_dispatch='2*n_jobs', 'n_jobs', None, cpu_count()-1, active_count()-1
            
            mean = np.mean(scores)
            standard_deviation = np.std(scores)
            
            #print(str(mean*100) + "\u00b1" + str(standard_deviation*100) + "\t", end='')
            print(str(round(mean*100, 2)) + "\u00b1" + str(round(standard_deviation*100, 2)) + "\t", end='') # simple rounding
            #print(str(int(np.floor(mean * 100))) + "\u00b1" + str(int(np.ceil(standard_deviation * 100))) + "\t", end='') # rounded to full %; down for mean, up for std
            #print("test\u00b1test\t", end='')

if __name__ == "__main__":
    main()