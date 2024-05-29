'''The entry point of our implementation.'''

# external imports
import argparse
import networkx as nx
import numpy as np
import pickle
import torch as th
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# internal imports
from src.preprocessing import max_node_dim, stack_adjacency, zero_pad
from src.models import GCN_graph, GCN_node



def one_hot_encoder(label: int, length: int) -> np.ndarray:
    '''returns one-hot vector according to given label integer'''
    zero_vector = np.zeros(length)
    zero_vector[label] = 1
    return zero_vector



def l2_renorm(real_list: list[float]) -> np.ndarray:
    '''returns vector renormalized to l2-norm of 1, used for normalizing 'attribute_labels' of nodes'''
    real_vector = np.array(real_list)
    l2_norm = np.linalg.norm(real_vector)
    return real_vector / l2_norm if l2_norm > 0 else real_vector



def data_tensor(graphs: list[nx.Graph], label_type: str, max_n_nodes: int) -> th.Tensor:
    '''returns one-hot encoded graph or node labels or l2-normalized 'node_attributes' as np-tensor in the format of the given dataset (w/ necessary zero-padding)'''
    if label_type == 'graph_label':  # for one-hot encoding graph labels
        # collect all available graph label values in dataset
        graph_labels = []
        for graph in graphs:
            label = graph.graph['label']
            if label not in graph_labels:
                graph_labels.append(label)

        # find min. and max. label values, create zero-vector
        min_label = min(graph_labels)
        one_hot_length = max(graph_labels) - min_label + 1

        return th.tensor(np.array([one_hot_encoder(graph.graph['label'] - min_label, one_hot_length) for graph in graphs]))
        #return th.tensor(np.array([graph.graph['label'] for graph in graphs])).type(th.long)  # try graph labels w/o one-hot encoding...

    elif label_type == 'node_label':  # for one-hot encoding node labels
        # collect all available node label values in dataset
        node_labels = []
        for graph in graphs:
            for node in graph.nodes(data=True):
                node_label = node[1]['node_label']
                if node_label not in node_labels:
                    node_labels.append(node_label)

        # find min. and max. label values, create zero-vector
        min_label = min(node_labels)
        one_hot_length = max(node_labels) - min_label + 1

        return th.tensor(np.array(
            [zero_pad(np.array(
                [one_hot_encoder(node[1]['node_label'] - min_label, one_hot_length)
                 for node in graph.nodes(data=True)]), [0], max_n_nodes)
             for graph in graphs]))

    else:  # for l2-normalizing node_attributes
        return th.tensor(np.array(
            [zero_pad(np.array(
                [l2_renorm(node[1]['node_attributes'])
                 for node in graph.nodes(data=True)]), [0], max_n_nodes)
             for graph in graphs]))



def train_eval_split(T: th.Tensor, split: int, split_length: int) -> tuple[th.Tensor, th.Tensor]:
    '''only for cross-validation on enzymes & nci1 (graph-lvl): returns train- & eval-split of given data tensor (T) for given [split] index'''
    T_splits = list(th.split(T, split_length, 0))
    T_eval = T_splits[split]
    del T_splits[split]  # removing eval_split from T_splits yields list of training splits
    T_train = th.cat(T_splits, 0)  # re-concatenate training splits into one training tensor
    return T_train, T_eval



def train_eval_resplit(graphs_train: list[nx.Graph], graphs_eval: list[nx.Graph], label_type: str, max_n_nodes: int) -> tuple[th.Tensor, th.Tensor]:
    '''only for citeseer & cora (node-lvl): concatenates resp. train- & eval-dataset (for X and Y separately) to find min. & max 'node_label' values over both, applies data_tensor() to one-hot encode/renormalize them, splits them back again, and returns them'''
    length_train = len(graphs_train)
    data_join = data_tensor(graphs_train + graphs_eval, label_type, max_n_nodes)  # run data_tensor() on joined train & eval graphs
    return data_join[: length_train], data_join[length_train :]  # split train & eval graphs again



def graph_lvl(graphs: list[nx.Graph], dataset: str, device: str, dropout_list: list[int], max_n_nodes: int, batch_size: int, n_epochs: int, n_splits: int = 10):
    '''training & testing node-level GCN model on ENZYMES & NCI1 with graph-level GCN'''
    print("for graph-level GCN...")
    """
    # in case dataset length is not divisable by n_splits
    length = len(graphs)  # number of graphs in dataset
    trunc_length = length - (length % n_splits)  # truncated dataset length
    split_length = int(trunc_length / n_splits)
    graphs = graphs[: trunc_length]  # truncate dataset to have number of graphs be an integer multiple of number of splits (n_splits), necessary for cross-validation
    """
    split_length = int(len(graphs) / n_splits)  # both datasets' lengths happen to be divisable by n_splits = 10

    if dataset == 'enzymes':
        X = th.cat((data_tensor(graphs, 'node_label', max_n_nodes), data_tensor(graphs, 'node_attributes', max_n_nodes)), -1)
    else:  #  elif dataset == 'nci1:
        X = data_tensor(graphs, 'node_label', max_n_nodes)
    Y = data_tensor(graphs, 'graph_label', max_n_nodes)
    A = stack_adjacency(graphs, max_n_nodes)

    #cross-validation
    accuracies = []  # collect accuracies from comparing predicted vs. true y-labels on test data for [n_rounds] rounds
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]

    for split in range(n_splits):
        print(f"\nSplit {split}:")
        #  produce train/eval-split for each dataset-derived tensor
        A_train, A_eval = train_eval_split(A, split, split_length)
        X_train, X_eval = train_eval_split(X, split, split_length)
        Y_train, Y_eval = train_eval_split(Y, split, split_length)

        dataset_train = TensorDataset(A_train, X_train, Y_train)  # match up training data according to graph
        #dataset_eval = TensorDataset(A_eval, X_eval, Y_eval)  # match up test data according to graph
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)  # shuffle & batch up matched training data
        #eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)  # shuffle & batch up matched test data
        accuracies.append(run_model(train_loader, A_eval, X_eval, Y_eval, input_dim, output_dim, len(Y_train), len(Y_eval), n_epochs, dropout_list, 'graph', device))

    return final_results(accuracies)



def node_lvl(graphs_train: list[nx.Graph], graphs_eval: list[nx.Graph], device: str, dropout_list: list[int], max_n_nodes: int, batch_size: int, n_epochs: int, n_rounds: int = 10):
    '''training & testing node-level GCN model on citeseer & cora with node-level GCN'''
    print("for node-level GCN...")
    # prepare data
    X_train, X_eval = train_eval_resplit(graphs_train, graphs_eval, 'node_attributes', max_n_nodes)
    Y_train, Y_eval = train_eval_resplit(graphs_train, graphs_eval, 'node_label', max_n_nodes)
    max_n_nodes = max(max_node_dim(graphs_train), max_node_dim(graphs_eval))
    A_train = stack_adjacency(graphs_train, max_n_nodes)
    A_eval = stack_adjacency(graphs_eval, max_n_nodes)

    dataset_train = TensorDataset(A_train, X_train, Y_train)  # match up training data according to graph
    #dataset_eval = TensorDataset(A_eval, X_eval, Y_eval)  # match up test data according to graph
    n_all_nodes_train = nx.number_of_nodes(graphs_train[0])
    n_all_nodes_eval = nx.number_of_nodes(graphs_eval[0])

    # train/eval cycles
    accuracies = []  # collect accuracies from comparing predicted vs. true y-labels on test data for [n_rounds] rounds
    input_dim = X_train.shape[-1]
    output_dim = Y_train.shape[-1]

    for round in range(n_rounds):
        print(f"\nRound {round}:")
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)  # shuffle & batch up matched training data
        #eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)  # shuffle & batch up matched test data
        accuracies.append(run_model(train_loader, A_eval, X_eval, Y_eval, input_dim, output_dim, n_all_nodes_train, n_all_nodes_eval, n_epochs, dropout_list, 'node', device))

    return final_results(accuracies)



def final_results(accuracies: list[float], digits: int = 2):
    '''prints mean +/- standard deviation over accuracies on given test data'''
    mean_accuracy = np.mean(accuracies) * 100
    std_accuracy = np.std(accuracies) * 100
    print(f"\n-> Mean \u00b1 StD of accuracy on test data (in %): {mean_accuracy} \u00b1 {std_accuracy}\n\n")
    return round(mean_accuracy, digits), round(std_accuracy, digits)  # return rounded accuracies for summary



def accuracy_sum(y_pred: th.Tensor, y_true: th.Tensor) -> float:
    '''accuracy (in general) = rate of all coincidences between true and predicted y-label values; here: sum up all coincidences between y_pred & y_true in given batch/dataset'''
    return (y_pred.argmax(-1) == y_true.argmax(-1)).type(th.float).sum().item()



def run_model(train_loader: DataLoader, A_eval: th.Tensor, X_eval: th.Tensor, Y_eval: th.Tensor, input_dim: int, output_dim: int, length_train: int, length_eval: int, n_epochs: int, dropout_list: list[int], model_type: str, device: str) -> float:
    '''create model & optimizer object, send everything necessary to [device], train & test on given data in several epochs per dataset/split, return accuracy on every given test dataset/split'''

    # construct GCN model (object) of given [model_type] w/ predefined initialization
    if model_type == 'graph':
        model = GCN_graph(input_dim, output_dim, dropout_list)
    else:  # elif model_type == 'node':
        model = GCN_node(input_dim, output_dim, dropout_list)

    model.to(device)  # move model to device
    model.train()  # switch model to training mode

    # construct optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)  # TODO optimizer parameters to be investigated, tested, chosen...

    # run gradient descent for [n_epochs]
    for epoch in range(n_epochs):
        #training phase
        accuracy_train = 0  # collect accuracies on training data itself

        for a, x, y_true in train_loader:
            # set gradients to zero
            optimizer.zero_grad()

            # move training batch to device
            x = x.to(device)
            y_true = y_true.to(device)
            a = a.to(device)

            # forward pass and loss
            y_pred = model(x, a)
            loss = F.cross_entropy(y_pred, y_true)

            # backward pass and sgd step
            loss.backward()
            optimizer.step()

            # update sum of coincindences on training data
            accuracy_train += accuracy_sum(y_pred, y_true)

        accuracy_train /= length_train  # accuracy on training data

        #testing phase
        model.eval()  # switch model to evaluation mode

        with th.no_grad():
            # move test data to device
            A_eval = A_eval.to(device)
            X_eval = X_eval.to(device)
            Y_eval = Y_eval.to(device)

            # evaluate forward fct. to predict y-values
            Y_pred = model(X_eval, A_eval)

        accuracy_eval = accuracy_sum(Y_pred, Y_eval) / length_eval  # accuracy on test data
        # loss.item() = loss after last batch in epoch
        print(f"epoch {epoch}: loss: {round(loss.item(), 4)}\t\tacc_train(%): {round(accuracy_train * 100, 4)}\t\tacc_eval(%): {round(accuracy_eval * 100, 4)}")

    return accuracy_eval  # return accuracy on test data after last epoch



def main(datasets: list[str]):
    '''chooses what to run according to parsed arguments, i.e. for which datasets and associated exercises'''
    print("---")

    #device = 'cpu'
    #device = 'cuda'
    device = ("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")  # choose by device priority
    print(f"Device: {device}\n")  # which device is being used for torch operations

    # TODO model parameters: to be tested, chosen, global vs. dataset-specific?

    # list of [dropout_list]s, indexing in which GCN single layer to add prior dropout layer for each dataset, inactive for now
    dropout_table = [[0, 0, 0, 0, 0],  # ENZYMES
                     [0, 0 ,0 ,0, 0],  # NCI1
                     [0, 0, 0],  # Citeseer
                     [0, 0, 0]]  # Cora

    # batch sizes for ENZYMES & NCI1
    batch_size_enzymes, batch_size_nci1 = 1, 137
    #1, 137 #100, 100 #60, 137 #540, 3699

    # number of epochs: find compromise on performance vs. runtime...
    n_epochs_enzymes, n_epochs_nci1, n_epochs_citeseer, n_epochs_cora = 50, 50, 150, 150
    #10, 10 #20, 20 #50, 50, 150, 150

    data_prepare = "Preparing dataset "
    end_results = []  # collect means & std.s of accuracies for each dataset

    for dataset in datasets:

        if dataset == 'enzymes':
            print(data_prepare + "ENZYMES ", end="")
            with open('datasets/ENZYMES/data.pkl', 'rb') as data:
                graphs = pickle.load(data)
            acc_mean, acc_std = graph_lvl(graphs, dataset, device, dropout_table[0], max_node_dim(graphs), batch_size_enzymes, n_epochs_enzymes)
            end_results.append([dataset + "\t", acc_mean, acc_std])

        elif dataset == 'nci1':
            print(data_prepare + "NCI1 ", end="")
            with open('datasets/NCI1/data.pkl', 'rb') as data:
                graphs = pickle.load(data)
            acc_mean, acc_std = graph_lvl(graphs, dataset, device, dropout_table[1], max_node_dim(graphs), batch_size_nci1, n_epochs_nci1)
            end_results.append([dataset + "\t", acc_mean, acc_std])

        elif dataset == 'citeseer':
            print(data_prepare + "Citeseer ", end="")
            with open('datasets/Citeseer_Train/data.pkl', 'rb') as data:
                graphs_train = pickle.load(data)
            with open('datasets/Citeseer_Eval/data.pkl', 'rb') as data:
                graphs_eval = pickle.load(data)
            acc_mean, acc_std = node_lvl(graphs_train, graphs_eval, device, dropout_table[2], max_node_dim(graphs_train + graphs_eval), 1, n_epochs_citeseer)
            end_results.append([dataset, acc_mean, acc_std])

        elif dataset == 'cora':
            print(data_prepare + "Cora ", end="")
            with open('datasets/Citeseer_Train/data.pkl', 'rb') as data:
                graphs_train = pickle.load(data)
            with open('datasets/Citeseer_Eval/data.pkl', 'rb') as data:
                graphs_eval = pickle.load(data)
            acc_mean, acc_std = node_lvl(graphs_train, graphs_eval, device, dropout_table[3], max_node_dim(graphs_train + graphs_eval), 1, n_epochs_cora)
            end_results.append([dataset + "\t", acc_mean, acc_std])

        else:
            print(f"Invalid dataset name '{dataset}' included in command line argument. Will be skipped.")

    print(f"---\nSummary:\nMean \u00b1 Standard Deviation of Accuracy Scores (rounded in %):\n")

    for end_result in end_results:
        print(f"{end_result[0]}\t{end_result[1]} \u00b1 {end_result[2]}")

    print("---")



if __name__ == "__main__":
    # configure parser
    parser = argparse.ArgumentParser()

    parser.add_argument('datasets', nargs='*', default=['enzymes', 'nci1', 'citeseer', 'cora'],
                        help="list of predefined *datasets* to be called by their resp. names ['enzymes', 'nci1', 'citeseer', 'cora'] (w/o quotes or brackets, separated by spaces only). Runs graph- or node-level evaluation (Ex.5/6) according to each called dataset. If left empty, defaults to calling all of them once in the above order. Names not included will be skipped.")  # positional argument

    args = parser.parse_args()  # parse from command line
    main(args.datasets)  # run w/ parsed arguments
