from src.dataset import Custom_Dataset
import pickle 

def load_data(type:str)->Custom_Dataset:
    '''loads dataset according to given type string'''
    with open(f'datasets/ZINC_{type}/data.pkl', 'rb') as data:
        graphs = pickle.load(data)

    return Custom_Dataset(graphs)

def test_one_hot():

    d = load_data("Train")