"""
functions for feature map
"""

from typing import Tuple, Union
from collections import defaultdict

import networkx as nx
import numpy as np
import torch

#code from strawberryfields
def resize(
    subgraph: list,
    graph: nx.Graph,
    min_size: int,
    max_size: int,
    node_select: Union[str, np.ndarray, list] = "uniform",
) -> dict:
    """Resize a subgraph to a range of input sizes, code from strawberryfields

    This function uses a greedy approach to iteratively add or remove nodes one at a time to an
    input subgraph to reach the range of sizes specified by ``min_size`` and ``max_size``.

    When growth is required, the algorithm examines all nodes from the remainder of the graph as
    candidates and adds the single node with the highest degree relative to the rest of the
    subgraph. This results in a graph that is one node larger, and if growth is still required,
    the algorithm performs the procedure again.

    When shrinking is required, the algorithm examines all nodes from within the subgraph as
    candidates and removes the single node with lowest degree relative to the subgraph.

    In both growth and shrink phases, there may be multiple candidate nodes with equal degree to
    add to or remove from the subgraph. The method of selecting the node is specified by the
    ``node_select`` argument, which can be either:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
    from the candidates with the highest weight (when growing) and lowest weight (when shrinking),
    settling remaining ties by uniform random choice.

    **Example usage:**

    >>> s = data.Planted()
    >>> g = nx.Graph(s.adj)
    >>> s = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> resize(s, g, 8, 12)
    {10: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    11: [11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    12: [0, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    9: [20, 21, 22, 24, 25, 26, 27, 28, 29],
    8: [20, 21, 22, 24, 25, 26, 27, 29]}

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size for subgraph to be resized to
        max_size (int): maximum size for subgraph to be resized to
        node_select (str, list or array): method of settling ties when more than one node of
            equal degree can be added/removed. Can be ``"uniform"`` (default), or a NumPy array or
            list containing node weights.

    Returns:
        dict[int, list[int]]: a dictionary of different sizes with corresponding subgraph
    """
    nodes = graph.nodes()
    subgraph = set(subgraph)
    node_select, w = _validate_inputs(subgraph, graph, min_size, max_size, node_select)

    starting_size = len(subgraph)

    if min_size <= starting_size <= max_size:
        resized = {starting_size: sorted(subgraph)}
    else:
        resized = {}

    if max_size > starting_size:

        grow_subgraph = graph.subgraph(subgraph).copy()

        while grow_subgraph.order() < max_size:
            grow_nodes = grow_subgraph.nodes()
            complement_nodes = nodes - grow_nodes

            degrees = np.array(
                [(c, graph.subgraph(list(grow_nodes) + [c]).degree()[c]) for c in complement_nodes]
            )
            degrees_max = np.argwhere(degrees[:, 1] == degrees[:, 1].max()).flatten()

            if node_select == "uniform":
                to_add_index = np.random.choice(degrees_max)
            elif node_select == "weight":
                weights = np.array([w[degrees[n][0]] for n in degrees_max])
                to_add_index = np.random.choice(np.where(weights == weights.max())[0])

            to_add = degrees[to_add_index][0]
            grow_subgraph.add_node(to_add)
            new_size = grow_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(grow_subgraph.nodes())

    if min_size < starting_size:

        shrink_subgraph = graph.subgraph(subgraph).copy()

        while shrink_subgraph.order() > min_size:
            degrees = np.array(shrink_subgraph.degree)
            degrees_min = np.argwhere(degrees[:, 1] == degrees[:, 1].min()).flatten()

            if node_select == "uniform":
                to_remove_index = np.random.choice(degrees_min)
            elif node_select == "weight":
                weights = np.array([w[degrees[n][0]] for n in degrees_min])
                to_remove_index = np.random.choice(np.where(weights == weights.min())[0])

            to_remove = degrees[to_remove_index][0]
            shrink_subgraph.remove_node(to_remove)

            new_size = shrink_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(shrink_subgraph.nodes())

    return resized

def _validate_inputs(
    subgraph: set,
    graph: nx.Graph,
    min_size: int,
    max_size: int,
    node_select: Union[str, np.ndarray, list] = "uniform",
) -> Tuple:
    """Validates input for the ``resize`` function.

    This function checks:
        - if ``subgraph`` is a valid subgraph of ``graph``;
        - if ``min_size`` and ``max_size`` are sensible numbers;
        - if ``node_select`` is either ``"uniform"`` or a NumPy array or list;
        - if, when ``node_select`` is a NumPy array or list, that it is the correct size and that
        ``node_select`` is changed to ``"weight"``.

    This function returns the updated ``node_select`` and a dictionary mapping nodes to their
    corresponding weights (weights default to unity if not specified).

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size for subgraph to be resized to
        max_size (int): maximum size for subgraph to be resized to
        node_select (str, list or array): method of settling ties when more than one node of
            equal degree can be added/removed. Can be ``"uniform"`` (default), or a NumPy array or
            list containing node weights.

    Returns:
        tuple[str, dict]: the updated ``node_select`` and a dictionary of node weights
    """
    if not subgraph.issubset(graph.nodes()):
        raise ValueError("Input is not a valid subgraph")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_size >= len(graph.nodes()):
        raise ValueError("max_size must be less than number of nodes in graph")
    if max_size < min_size:
        raise ValueError("max_size must not be less than min_size")

    if isinstance(node_select, (list, np.ndarray)):
        if len(node_select) != graph.number_of_nodes():
            raise ValueError("Number of node weights must match number of nodes")
        w = {n: node_select[i] for i, n in enumerate(graph.nodes)}
        node_select = "weight"
    else:
        w = {n: 1 for i, n in enumerate(graph.nodes)}
        if node_select != "uniform":
            raise ValueError("Node selection method not recognized")

    return node_select, w


def search_subgpraph(samples: list,
                    graph: nx.Graph,
                    min_size: int,
                    max_size: int):
    """Get the densest subgraph with size in [min_size, max_size],
        using classical algorithm with samples from GBS
    """
    dic_list = defaultdict(list)
    for i in range(len(samples)):
        temp= samples[i]
        num = 1
        for key in temp.keys():
            if num < 50: # only need 50 samples
                idx = torch.nonzero(torch.tensor(key)).squeeze()
                r = resize(idx.tolist(), graph, min_size=min_size, max_size=max_size)
                for j in range(min_size, max_size+2, 2):
                    density = nx.density(graph.subgraph(r[j]))
                    temp_value = (r[j], np.round(density, 5))
                    if temp_value not in dic_list[j]:
                        dic_list[j].append(temp_value)
                num = num + 1
    return dic_list
