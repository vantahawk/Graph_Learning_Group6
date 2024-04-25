import aenum

#"closed_walk", "shortest_path", "graphlet", "wl", "subtree", "rooted_kernel"
class KernelName(aenum.StrEnum):
    closed_walk = "closed_walk"
    # shortest_path = "shortest_path"
    graphlet = "graphlet"
    wl = "wl"
    # subtree = "subtree"
    # rooted_kernel = "rooted_kernel"