import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import List, Dict

kernelDict = {
    "closed_walk": "Closed Walk",
    "graphlet": "Graphlet",
    "wl": "Weisfeiler-Lehman",
}


def generate_plots(kernel:str|List[str]):

    base_path:str = os.path.join("out", "svc_KERNEL.csv")

    kernels:List[str] = []

    if isinstance(kernel, str):
        kernels = [kernel]
    else:
        kernels = kernel
    
    #csv looks like this, best_params is the dict of the params the best svm had, for each dataset
    # dataset,          best_params_<dataset1>, best_params_<dataset2>, ...
    # dataset1_mean,    Acc1,                   /, ...
    # dataset1_std,     Acc1,                   /, ...
    # dataset2_mean,    /,                      Acc2, ...
    # dataset2_std,     /,                      Acc2, ...
    # ...

    
    dfs:Dict[str, pd.DataFrame] = {}

    for k in kernels:
        df:pd.DataFrame = pd.read_csv(base_path.replace("KERNEL", k), index_col=0)
        #fill nan with 0
        df.fillna(0, inplace=True)
        dfs |= {k:df}
        #
    
    plot_df_mean = pd.DataFrame(columns=[dataset.removesuffix("-mean") for dataset in dfs["closed_walk"].index if dataset.endswith("mean")], index=kernels)
    plot_df_std = pd.DataFrame(columns=[dataset.removesuffix("-mean") for dataset in dfs["closed_walk"].index if dataset.endswith("mean")], index=kernels)

    for k in kernels:
        for dataset in dfs[k].index:
            if dataset.endswith("mean"):
                plot_df_mean.loc[k, dataset.removesuffix("-mean")] = dfs[k].loc[dataset, :].sum()
            elif dataset.endswith("std"):
                plot_df_std.loc[k, dataset.removesuffix("-std")] = dfs[k].loc[dataset, :].sum()
            else:
                print("wtf")
                continue

    datasets = plot_df_mean.columns

    #plot the data
    #we want one figure with columns for each kernel, and other sets of columns for each dataset. Mean is the column and std is the error bar
    #choose a nice style
    plt.style.use("seaborn-v0_8-deep")
    fig, ax = plt.subplots()

    xs = list(np.arange(0, 1.3*len(datasets), 1.3))
    width = 0.35

    #use colors for the different kernels
    for i, k in enumerate(kernels):
        ax.bar([x+ (i-1)*width for x in xs], plot_df_mean.loc[k, :], width-0.1, label=kernelDict[k], yerr=plot_df_std.loc[k, :], capsize=5)

    ax.yaxis.grid(True)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    ax.set_title("Accuracy by dataset and kernel")

    ax.set_xticks(xs)
    ax.set_xticklabels(datasets)

    ax.legend()

    plt.show()



generate_plots(["closed_walk", "graphlet", "wl"])