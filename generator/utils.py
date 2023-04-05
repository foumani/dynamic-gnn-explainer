import random

import networkx as nx
import numpy as np


def perturb(G: nx.Graph, n: int = None, p: float = None) -> nx.Graph:
    assert n is None or 0 < n
    assert p is None or 0.0 <= p <= 1.0
    if all(v is None for v in {n, p}):
        raise ValueError("Expected either n or p args")
    if all(v is not None for v in {n, p}):
        raise ValueError("One of n or p is required")

    if p is not None:
        n = max(int(G.number_of_nodes() * p), 1)

    G = G.copy()
    for _ in range(n):
        while True:
            u = G.nodes(np.random.randint(1, G.number_of_nodes()))
            v = G.nodes(np.random.randint(1, G.number_of_nodes()))
            if (not G.has_edge(u, v)) and (u != v): break
        G.add_edge(u, v)
    return G


def dynamic_perturb(graphs: list, p: float) -> list:
    # TODO: complete the method, then use it in DynamicGraphGenerator
    graphs = [graph.copy() for graph in graphs]
    n = p * len(graphs[0])
    T = len(graphs)
    for _ in range(n):
        while True:
            t_change = random.randint(0, T - 1)  # included start
            t_revert = random.randint(0, T)  # excluded end
            t_change, t_revert = min(t_change, t_revert), max(t_change, t_revert)
            if t_change - t_revert > 1: break
        while True:
            u = random.choice(list(graphs[t_change]))
            v = random.choice(list(graphs[t_change]))
            if graphs[t_change].has_edge(u, v) or u == v: continue
            for t in range(t_change, t_revert):
                graphs[t].add_edge(u, v)
            break
    return graphs
