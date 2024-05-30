import torch as th
from torch_scatter import scatter_sum



class Sum_Pooling(th.nn.Module):
    '''module for sparse sum pooling layer, reduces network from node- to graph-level'''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
        '''forward fct. for sparse sum pooling using scatter_sum() according to batch_idx, i.e. summing each node embedding in a batch graph towards its original graph'''
        return scatter_sum(x, batch_idx, dim=0)
