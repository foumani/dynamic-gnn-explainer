import pandas as pd

def graphs_to_csv(graphs, roles):
    classes = pd.DataFrame(columns=["txId", "class"])
    roles_series = pd.Series(roles)
    nodes_series = pd.Series([i for i in range(len(roles))])
    classes["txId"] = nodes_series
    classes["class"] = roles_series
    classes.to_csv("elliptic_txs_classes.csv")

    #%%
edgelist = pd.DataFrame(columns=["txId1", "txId2", "timestep"])
    a, b, times = [], [], []
    for t in range(len(graphs)):
        graph = graphs[t]
        for e in graph.edges:
            a.append(e[0])
            b.append(e[1])
            b.append(e[1])
            a.append(e[1])
            times.append(t)
            times.append(t)

    a_series = pd.Series(a)
    b_series = pd.Series(b)
    t_series = pd.Series(times)
    edgelist["txId1"] = a_series
    edgelist["txId2"] = b_series
    edgelist["timestep"] = t_series
    edgelist.to_csv("elliptic_txs_edgelist_timed.csv")

    #%%
nodetime = pd.DataFrame(columns=["txId", "timestep"])
    nodes_times = pd.Series([0 for _ in range(len(roles))])
    nodes = pd.Series([i for i in range(len(roles))])
    nodetime["txId"] = nodes
    nodetime["timestep"] = nodes_times
    nodetime.to_csv("elliptic_txs_nodetime.csv")