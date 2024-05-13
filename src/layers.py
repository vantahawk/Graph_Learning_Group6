from torch.nn import Module, Parameter, init as init_params, AvgPool2d
from torch import Tensor, empty as empty_tensor, bmm as batch_matmul, zeros
from torch.nn.functional import pad as pad_tensor
from typing import Literal, Tuple, List
from .utils import Shape

Padding = Literal['zeroes', 'reflect', 'wrap']

class GeneralGCNLayer(Module):

    def __init__(self, embed_dim:int, out_dim:int, shape:Shape = None, use_bias:bool=False):#
        super().__init__()

        if shape != None:
            assert len(shape) == 2, "The given shape does not have sizes for both dimensions : embed_dimension, out_dimension"
            embed_dim, out_dim = shape

        self.W:Tensor = Parameter(empty_tensor(embed_dim, out_dim))
        init_params.kaiming_normal_(self.W)

        if use_bias:
            self.b:Tensor = Parameter(empty_tensor(1, out_dim))
            init_params.zeros_(self.b)
        self.use_bias = use_bias

    def forward(self, A:Tensor, H:Tensor):
        """A forward pass of the GCN layer. Gets the last embedding of the node features and returns a new one.
        Args:
            - A (Tensor): The normalized Adjacencymatrix, must be of shape:
                (n_nodes, n_nodes)
            - H (Tensor): the last layers computed graph embedding, must be of shape: (batch_size, n_nodes, embed_dim).
        
        Returns a Tensor H'.
        """
        batch_size = H.size(0)
        #Expand the view to make the tensors virtually larger
        A = A.unsqueeze(0).expand(batch_size, -1, -1)
        W = self.W.unsqueeze(0).expand(batch_size, -1, -1)

        #compute the KQV product of the tensors
        H_prime:Tensor = batch_matmul(A, H)
        H_prime = batch_matmul(H_prime, W)

        if self.use_bias:
            #gets broadcasted automatically? #TODO figure out if this actually happens. But it should.
            H_prime = H_prime + self.b.unsqueeze(0)

        return H_prime

class Pool2D(Module):

    def __init__(self, size:int=3, stride:int=1, padding:Padding='zeroes'):
        
        self.pool_layer = AvgPool2d(size, stride)
        self.padding:Padding = padding
        self.pad_size:Tuple[int, int, int, int] = tuple(size//2 for _ in range(4))

    def forward(self, x:Tensor)->Tensor:

        x = pad_tensor(x, self.pad_size, mode=self.padding)
        x = self.pool_layer(x)
        return x

PoolType = Literal["sum"]
class PoolNodeEmbeddings(Module):
    def __init__(self, type:PoolType|List[PoolType]="sum"):
        """Pools the Node Embeddings along the first axis (1): the nodes -> resulting vector has the length of the inputted embedding. Expects a batch."""
        self.pool_types:List[PoolType] = type if isinstance(type, list) else [type]

    def forward(self, H:Tensor)->Tensor:
        """Returns a vector where the entries are the summed components over all nodes."""
        # R:Tensor = zeros((H.shape[0], H.shape[2]))

        # #future proof, maybe we want to have a more special pooling.
        # for pool_type in self.pool_types:
        #     match pool_type:
        #         case "sum":
        #             R += H.sum(1)#dim 1 is the node dimension

        # return R
        return H.sum(1)
