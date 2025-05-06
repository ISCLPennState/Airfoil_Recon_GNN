
from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean
from torch_geometric.data import Batch

from torch_geometric.utils import softmax, remove_self_loops, coalesce

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import uniform


from torch_geometric.nn.pool.consecutive import consecutive_cluster


# avg pooling 
def avg_pool_mod(cluster, x, edge_index, edge_attr, batch, pos):
    # Makes cluster indices consecutive, to allow for scatter operations 
    # -- cluster = [0, 0, 4, 3, 3, 4]
    # -- cons_cluster = [0, 0, 2, 1, 1, 2]
    cluster, perm = consecutive_cluster(cluster)

    # Pool node attributes 
    # x_pool = None if x is None else _avg_pool_x(cluster, x)
    x_pool = None if x is None else scatter(x, cluster, dim=0, dim_size=None, reduce='mean')

    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)

    # Pool batch 
    batch_pool = None if batch is None else batch[perm]

    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)

    return x_pool, edge_index_pool, edge_attr_pool, batch_pool, pos_pool, cluster, perm



def avg_pool_mod_no_x(cluster, edge_index, edge_attr, batch, pos):
    # Makes cluster indices consecutive, to allow for scatter operations 
    # -- cluster = [0, 0, 4, 3, 3, 4]
    # -- cons_cluster = [0, 0, 2, 1, 1, 2]
    cluster, perm = consecutive_cluster(cluster)

    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)

    # Pool batch 
    batch_pool = None if batch is None else batch[perm]

    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)

    return edge_index_pool, edge_attr_pool, batch_pool, pos_pool, cluster, perm


# Edge pooling
def pool_edge_mean(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1) 
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
    return edge_index, edge_attr
