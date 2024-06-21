'''entry point of our full implementation'''
# external imports
import argparse
import pickle
import torch as th

# internal imports
from node_class import node_class
from link_pred import link_pred

def main(datasets: list[str], p: float, q: float, l: int, l_ns: int, dim: int,  # main parameters, see sheet
         special: bool, n_batches: int, batch_size: int, return_train_score: bool,  # extra parameters
         k: int, eval_share: float, lr: float) -> None:
    '''runs node classification & link prediction on the predefined datasets & parsed parameters, see argparse-arguments for more info'''
    print("---")
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    if special:
        datasets_valid, datasets_invalid = [], []
        for dataset in set(datasets):
            if dataset in ['Citeseer', 'Cora', 'Facebook', 'PPI']:
                datasets_valid.append(dataset)
            else:
                datasets_invalid.append(dataset)
        if datasets_invalid != []:
            print(f"Invalid dataset names {datasets_invalid} will be skipped.")
        for dataset in datasets_valid:
            print(f"\nSpecial run for dataset: {dataset}")
            with open('datasets/' + dataset + '/data.pkl', 'rb') as data:
                graph = pickle.load(data)[0]
            if dataset in ['Citeseer', 'Cora']:
                print(f"Node Classification w/ parameters:\np = {p}, q = {q}, l = {l}, l_ns = {l_ns}, dim = {dim}, n_batches = {n_batches}, batch_size = {batch_size}, k = {k}, lr = {lr}")
                results = node_class(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score, k, lr)
                print(f"Mean \u00b1 StD of Accuracy Scores (rounded in %) for:")
                for mode, mean, std in results:
                    print(f"{mode}:\t{round(mean * 100, 2)} \u00b1 {round(std * 100, 2)}")
            else:
                print(f"Link Prediction w/ parameters:\np = {p}, q = {q}, l = {l}, l_ns = {l_ns}, dim = {dim}, n_batches = {n_batches}, batch_size = {batch_size}, k = {k}, eval_share = {eval_share}, lr = {lr}")
                results = link_pred(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score, k, eval_share, lr)
                print(f"Mean \u00b1 StD of Scores (rounded in %) for:")
                for mode, mean, std in results:
                    print(f"{mode}:\t{round(mean * 100, 2)} \u00b1 {round(std * 100, 2)}")
    else:
        print("Default Run\n\nNode Classification\n")
        for dataset in ['Citeseer', 'Cora']:
            print(f"Dataset: {dataset}")
            with open('datasets/' + dataset + '/data.pkl', 'rb') as data:
                graph = pickle.load(data)[0]
            for pp, qq in [(1.0, 1.0), (0.1, 1.0), (1.0, 0.1)]:
                results = node_class(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
                print(f"\nMean \u00b1 StD of Accuracy Scores (rounded in %) for p = {pp}, q = {qq}:")
                for mode, mean, std in results:
                    print(f"{mode}:\t{round(mean * 100, 2)} \u00b1 {round(std * 100, 2)}")
            print("\n")
        print("\nLink Prediction\n")
        for dataset in ['Facebook', 'PPI']:
            print(f"Dataset: {dataset}")
            with open('datasets/' + dataset + '/data.pkl', 'rb') as data:
                graph = pickle.load(data)[0]
            results = link_pred(graph, p, q, l, l_ns, dim, n_batches, batch_size, device, return_train_score)
            print(f"\nMean \u00b1 StD of Scores (rounded in %) for p = q = {p}:")
            for mode, mean, std in results:
                print(f"{mode}:\t{round(mean * 100, 2)} \u00b1 {round(std * 100, 2)}")
            print("\n")
    print("---")



if __name__ == "__main__":
    # configure parser
    parser = argparse.ArgumentParser()

    parser.add_argument('datasets', nargs='*', default=['Citeseer', 'Cora', 'Facebook', 'PPI'],
                        help="List of predefined [datasets] to be called by their resp. names ['Citeseer', 'Cora', 'Facebook', 'PPI'] (w/o quotes or brackets, separated by spaces only). Runs node classification (Ex.6) of each called dataset. If left empty, defaults to calling all of them once in the above order. Names not included will be skipped")  # positional argument

    parser.add_argument('-s', '--special', action='store_true',
                        help="Flag to run a special instance of program, where the parsed set of datasets is evaluated (node classification (Ex.3) for Citeseer & Cora, link prediction (Ex.4) for Facebook & PPI) w/ a set of parsed parameters. Any parameter not set defaults to the values defined below. If the flag is not set, the default setting of the program will be run instead, w/ the fixed parameters prescribed in the exercise sheet, and w/ 3 turns of node classification on Citeseer & Cora for [(p=q=1.0), (p=0.1, q=1.0), (p=1.0, q=0.1)], as well as 1 turn of link prediction for p=q=1.0")  # optional argument

    # parameters as optional arguments
    parser.add_argument('-p', '--p', nargs='?', default=1.0, const=1.0, type=float, help="parameter p for pq-walk, ~BFS")
    parser.add_argument('-q', '--q', nargs='?', default=1.0, const=1.0, type=float, help="parameter q for pq-walk, ~DFS")
    parser.add_argument('-l', '--l', nargs='?', default=5, const=5, type=int, help="parameter l for pq-walk: walk length")
    parser.add_argument('-lns', '--l_ns', nargs='?', default=5, const=5, type=int, help="parameter l for pq-walk: number of negative node samples")
    parser.add_argument('-d', '--dim', nargs='?', default=128, const=128, type=int, help="feature dimension of node2vec embedding tensor X")
    parser.add_argument('-nb', '--n_batches', nargs='?', default=100, const=100, type=int, help="number of pq-walk batches")
    parser.add_argument('-bs', '--batch_size', nargs='?', default=1000, const=1000, type=int, help="number of pq-walks per batch")
    parser.add_argument('-rts', '--return_train_score', action='store_true',
                        help="additionally evaluate trained model on training data and return resp. scores")
    parser.add_argument('-k', '--k', nargs='?', default=5, const=10, type=int, help="number of splits/rounds to run for each evaluation")
    parser.add_argument('-es', '--eval_share', nargs='?', default=0.2, const=0.2, type=float,
                        help="share of eval edges to be sampled for link prediction")
    parser.add_argument('-lr', '--lr', nargs='?', default=0.01, const=0.001, type=float, help="learning rate for Adam optimizer in node2vec")

    args = parser.parse_args()  # parse from command line or pass string manually w/ .split()
    main(args.datasets, args.p, args.q, args.l, args.l_ns, args.dim,  # main parameters, see sheet
         args.special, args.n_batches, args.batch_size, args.return_train_score,  # extra parameters
         args.k, args.eval_share, args.lr)  # run w/ parsed arguments
