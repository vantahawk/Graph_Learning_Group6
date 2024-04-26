import aenum
from .base import KerneledGraph
from .closed_walk import ClosedWalkKernel
from .graphlet import GraphletKernel
from .wl import WLKernel
from typing import Type

#"closed_walk", "shortest_path", "graphlet", "wl", "subtree", "rooted_kernel"
class KernelName(aenum.StrEnum):
    closed_walk = "closed_walk"
    # shortest_path = "shortest_path"
    graphlet = "graphlet"
    wl = "wl"
    # subtree = "subtree"
    # rooted_kernel = "rooted_kernel"

KernelMap:dict[str, Type] = {
    "closed_walk": ClosedWalkKernel,
    # "shortest_path": ShortestPath,
    "graphlet": GraphletKernel,
    "wl": WLKernel,
    # "subtree": Subtree,
    # "rooted_kernel": RootedKernel
}