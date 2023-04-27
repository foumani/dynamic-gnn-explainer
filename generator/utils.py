import random

import networkx as nx
import numpy as np
import pandas as pd


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


def save_graphs(graphs: list, roles: list):
    # saving
    classes = pd.DataFrame(columns=["txId", "class"])
    roles_series = pd.to_numeric(pd.Series(roles), downcast="float")
    nodes_series = pd.to_numeric(pd.Series([i for i in range(len(roles))]), downcast="float")
    classes["txId"] = nodes_series
    classes["class"] = roles_series - 1
    classes.to_csv("elliptic_txs_classes.csv", index=False)
    print(classes)
    # saving edgelist of the graph
    edgelist = pd.DataFrame(columns=["txId1", "txId2", "timestep"])
    a, b, times = [], [], []
    for t in range(len(graphs)):
        graph = graphs[t]
        for e in graph.edges:
            a.append(e[0])
            b.append(e[1])
            times.append(t)

    a_series = pd.to_numeric(pd.Series(a))
    b_series = pd.to_numeric(pd.Series(b))
    t_series = pd.to_numeric(pd.Series(times), downcast="float")
    edgelist["txId1"] = a_series
    edgelist["txId2"] = b_series
    edgelist["timestep"] = t_series
    edgelist.to_csv("elliptic_txs_edgelist_timed.csv", index=False)
    edgelist

    # saving time of existence for each node
    nodetime = pd.DataFrame(columns=["txId", "timestep"])
    nodes, times = [], []
    for t in range(len(graphs)):
        graph = graphs[t]
        nodes.extend([i for i in range(len(roles))])
        times.extend([t for _ in range(len(roles))])
    nodes_times = pd.Series(times)
    nodes = pd.Series(nodes)
    nodetime["txId"] = nodes
    nodetime["timestep"] = nodes_times
    nodetime.to_csv("elliptic_txs_nodetime.csv", index=False)
    nodetime

    # saving features (one hot) for each node
    nodes, times = [], []
    for t in range(len(graphs)):
        graph = graphs[t]
        nodes.extend([i for i in range(len(roles))])
        times.extend([t for _ in range(len(roles))])
    nodes = pd.Series(nodes)
    features = pd.get_dummies(nodes, dtype="float")
    features.insert(0, "nodes", nodes, True)
    features.insert(1, "times", times, True)
    features.to_csv("elliptic_txs_features.csv", header=False, index=False)
