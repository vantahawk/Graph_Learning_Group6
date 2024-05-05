"""The entry point of our graph kernel implementation."""
#internal imports
from kernels import KernelName, KerneledGraph
from decorators import parseargs

#external imports
import importlib
from typing import List, Dict, Any, Union

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import cross_val_score, RandomizedSearchCV#, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
import psutil
import numpy as np
import pickle, os
import pandas as pd

SVMClassifier = Union[SVC, LinearSVC, NuSVC]

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
        "k":5,
        "m":1000
    },
    "wl": {
        "k":4
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
    save_results={
        "default":False,
        "type":bool,
        "help":"Whether to save the results to a file.",
        "flags":["sr", "save-results"]
    },
    classifier_type={
        "default":"SVC",
        "type":str,
        "help":"The type of classifier to use, one of: \"SVC\", \"LinearSVC\", \"NuSVC\"",
        "flags":["ct", "classifier-type", "clf-type"]
    },
    __description="The entry point of our graph kernel implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
     __help=True
)
def main(kernelname:KernelName, cv:int, save_results:bool, classifier_type:str):
    print(f"Using kernel \"{kernelname}\".")
    if kernelname not in kernelNameDict:
        print(f"Unknown kernel: \"{kernelname}\". Choose one of: {', '.join(kernelNameDict.keys())}.")
        exit(1)
    
    if classifier_type not in ["SVC", "LinearSVC", "NuSVC"]:
        print(f"Unknown classifier type: \"{classifier_type}\". Choose one of: \"SVC\", \"LinearSVC\", \"NuSVC\".")
        exit(1)

    kernel = importlib.import_module(f"kernels.{kernelname}")
    
    #SVM-modules:
    classifier:SVMClassifier = None
    search_space:Dict[str, List[Any]] = {}
    match classifier_type:
        case "SVC":
            classifier = SVC()
            search_space = {
                'C': [0.1, 0.2, 0.5, 1, 2, 5, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale'],
                'coef0': [0.0, 0.1, 0.2, 0.5, 1.0],
                'tol': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                'cache_size': [200, 1000, 2000, 5000, 10000],
                'max_iter': [-1],
                'decision_function_shape': ['ovr'],
                'break_ties': [True, False]
            }

        case "LinearSVC":
            # classifier = LinearSVC()
            # search_space = {
            #     'C': [0.1, 0.2, 0.5, 1, 2, 5, 10],
            #     'penalty': ['l2'],
            #     'loss': ['hinge', 'squared_hinge'],
            #     'dual': [True],
            #     'tol': [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            #     'multi_class': ['ovr'],
            #     'fit_intercept': [True, False],	
            #     'max_iter': [10000, 20000, 50000],
            # }
            print("LinearSVC works badly, thus we don't use it.")
            exit(3)

        case "NuSVC":
            classifier = NuSVC()
            search_space = {
                'nu': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale'],
                'coef0': [0.0, 0.1, 0.2, 0.5, 1.0],
                'tol': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                'cache_size': [200, 1000, 2000, 5000, 10000],
                'max_iter': [-1],
                'decision_function_shape': ['ovr'],
                'break_ties': [True, False]
            }
    

    classifier = make_pipeline(StandardScaler(with_mean=kernelname!="wl"), classifier) # use std-normalization of kernel vector data
    classifier = RandomizedSearchCV(classifier, {(f"{classifier_type.lower()}__"+k):v for k,v in search_space.items()},n_iter=500, n_jobs=psutil.cpu_count(), cv=cv, verbose=0, scoring='accuracy', error_score='raise',  pre_dispatch='2*n_jobs')
    
    
    outpath:str = f"out/{classifier_type.lower()}_{kernelname}.csv"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    outframe:pd.DataFrame = pd.DataFrame(
        data = np.full((len(datasets)*2, len(datasets)), np.nan, dtype=float), #np.full((len(datasets)*2, len(search_space)+1), np.nan, dtype=float
        index=[dataset+ "-mean" for dataset in datasets]+[dataset +"-std" for dataset in datasets],
        columns = datasets
    )

    for dataset in datasets:
        dataset_path:str = f'datasets/{dataset}/data.pkl'
        print("Dataset:", dataset)
        kerneledGraphs: List[KerneledGraph] = getattr(kernel, kernelNameDict[kernelname]).readGraphs(graphPath=dataset_path, **defaultKernelArgs[kernelname])


        with open(dataset_path, 'rb') as data:
            graphs = pickle.load(data)
            graph_labels = [graph.graph['label'] for graph in graphs]
            del graphs

            classifier.fit(kerneledGraphs, graph_labels)

        if save_results:
            #have the params be the columns and the dataset be the index, the best params get a 2nd column called "best"
            #save the mean and std as the values
            # for p, params in enumerate(classifier.cv_results_['params']):
            #     pure_params = {k.removeprefix("svc__"):v for k,v in params.items()}
            #     outframe.rename(columns={p+1:str(pure_params)}, inplace=True)
            #     outframe.loc[dataset+"-mean",str(pure_params)] = classifier.cv_results_["mean_test_score"][p]
            #     outframe.loc[dataset+"-std", str(pure_params)] = classifier.cv_results_["std_test_score"][p]

            outframe.loc[dataset+"-mean", dataset] = classifier.cv_results_["mean_test_score"][classifier.best_index_]
            outframe.loc[dataset+"-std",dataset] = classifier.cv_results_["std_test_score"][classifier.best_index_]

            outframe.rename(columns={dataset:str({k.removeprefix("svc__"):v for k,v in classifier.best_params_.items()})}, inplace=True)

        else:
            print("\n")
            mean = classifier.cv_results_["mean_test_score"][classifier.best_index_]
            standard_deviation = classifier.cv_results_["std_test_score"][classifier.best_index_]
            print(f"Accuracy: {round(mean*100, 2)}\u00b1{round(standard_deviation*100, 2)}")


    if save_results:
        outframe.to_csv(outpath)
        print(f"Results saved to: \"{outpath}\"")


if __name__ == "__main__":
    main()