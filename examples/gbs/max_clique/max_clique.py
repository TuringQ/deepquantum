"""
functions for max_clique problem
"""

from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from strawberryfields.apps import clique

def clique_shrink(samples, graph, node_select= "uniform") -> list:
    """Shrinks an input subgraph until it forms a clique, code from strawberryfields.
    """
    small_clique = [ ]
    max_node = 0
    for key in samples.keys():
        idx = torch.nonzero(torch.tensor(key)).squeeze()
        subgraph = idx.tolist()
        if not set(subgraph).issubset(graph.nodes):
            raise ValueError("Input is not a valid subgraph")

        if isinstance(node_select, (list, np.ndarray)):
            if len(node_select) != graph.number_of_nodes():
                raise ValueError("Number of node weights must match number of nodes")
            w = {n: node_select[i] for i, n in enumerate(graph.nodes)}
            node_select = "weight"

        subgraph = graph.subgraph(subgraph).copy()  # A copy is required to be able to modify the
        # structure of the subgraph
        while not clique.is_clique(subgraph):
            degrees = np.array(subgraph.degree)
            degrees_min = np.argwhere(degrees[:, 1] == degrees[:, 1].min()).flatten()

            if node_select == "uniform":
                to_remove_index = np.random.choice(degrees_min)
            elif node_select == "weight":
                weights = np.array([w[degrees[n][0]] for n in degrees_min])
                to_remove_index = np.random.choice(np.where(weights == weights.min())[0])
            else:
                raise ValueError("Node selection method not recognized")

            to_remove = degrees[to_remove_index][0]
            subgraph.remove_node(to_remove)
        if len(subgraph.nodes())>=max_node: #只保留找到较大的团作为起点
            max_node = len(subgraph.nodes())
            small_clique.append(sorted(subgraph.nodes()))

    return small_clique

def plot_subgraph(graph, subgraph_idx):
    """plot the subgraph in graph G"""

    sub_g = graph.subgraph(subgraph_idx).copy()
    edge_list = list(sub_g.edges)

    pos = nx.spring_layout(graph)
    nx.draw(graph,
            pos,
            with_labels=True,
            node_color='gray',
            edge_color="gray",
            node_size=200,
            font_size=10)
    nx.draw_networkx_edges(graph, pos,edgelist=edge_list,edge_color='blue')
    nx.draw_networkx_nodes(graph, pos, subgraph_idx, node_color='dodgerblue')
    plt.show()