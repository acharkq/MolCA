import torch
import numpy as np


def drop_nodes(data, rate=0.1):
    """
    Randomly dropping certain ratio of nodes.
    For those nodes to be dropped, remove all their edges by the following statements:
    adj[drop_node_idx, :] = 0, adj[:, drop_node_idx] = 0.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: drop node rate
    :return: output (class: torch_geometric.data.Data)
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * rate)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
#     print(idx_drop)
    edge_index = data.edge_index.numpy()
    ori_edge_index = edge_index.T.tolist()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    aft_edge_index = edge_index.numpy().T.tolist()
    keep_idx = []
    for idx, each in enumerate(ori_edge_index):
        if each in aft_edge_index:
            keep_idx.append(idx)
    data.edge_attr = data.edge_attr[keep_idx, :]
#     print(list(set(range(node_num))-set(idx_drop.tolist())))
#     data.x = data.x[list(set(range(node_num))-set(idx_drop.tolist())),:]

    return data


def permute_edges(data, rate=0.1, only_drop=True):
    """
    Randomly adding and dropping certain ratio of edges.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: add or drop edge rate
    :param only_drop: if True, only drop edges; if False, not only add but also drop edges
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * rate)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    idx_add = [[idx_add[n, 0], idx_add[n, 1]] for n in range(permute_num) if
               [idx_add[n, 0], idx_add[n, 1]] not in edge_index.tolist()]
    # print(idx_add)
    if not only_drop and idx_add:
        edge_index = np.concatenate(
            (edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)], idx_add), axis=0)
    else:
        edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def subgraph(data, rate=0.8):
    """
    Samples a subgraph using random walk.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: rate
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * rate)

    edge_index = data.edge_index.numpy()
    ori_edge_index = edge_index.T.tolist()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
#     print(idx_sub)
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])
#     print(idx_neigh)

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > 1.5 * node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))
#         print(idx_neigh)
#     print(idx_sub)
#     print(idx_neigh)

    idx_drop = [n for n in range(node_num) if n not in idx_sub]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    aft_edge_index = edge_index.numpy().T.tolist()
    keep_idx = []
    for idx, each in enumerate(ori_edge_index):
        if each in aft_edge_index:
            keep_idx.append(idx)
    data.edge_attr = data.edge_attr[keep_idx,:]

    return data


def mask_nodes(data, rate=0.1):
    """
    Randomly masking certain ratio of nodes.
    For those nodes to be masked, replace their features with vectors sampled in a normal distribution.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: mask node rate
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * rate)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                    dtype=torch.float32)

    return data
