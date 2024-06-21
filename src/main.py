import argparse
from link_pred import main as predict_links
from node_class import main as classify_nodes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Graph Tasks using Node2Vec.")
    parser.add_argument("--task", "-t", type=str, default="link", help="The type of task to perform. One of \"link\", \"node\".")
    parser.add_argument("--dataset", "-d", type=str, default="Facebook", help="The dataset to use. If task is link: one of [\"Facebook\", \"PPI\"], else: one of [\"Citeseer\", \"Cora\"]")

    args = parser.parse_args()

    task = args.task
    dataset = args.dataset

    if task == "link":
        if not dataset in ["Facebook", "PPI"]:
            raise ValueError("Wrong dataset for the task, choose either \"Facebook\" or \"PPI\".")
        print(f"Doing Link Prediction on Dataset {dataset}.")
        predict_links(dataset)

    elif task == "node":
        if not dataset in ["Citeseer", "Cora"]:
            raise ValueError("Wrong dataset for the task, choose either \"Citeseer\" or \"Cora\".")
        print(f"Doing Node Classification on Dataset {dataset}.")
        classify_nodes(dataset)

    else:
        raise ValueError("Wrong task, you may only choose between \"link\" and \"node\" for link prediction and node classification, respectively.")
