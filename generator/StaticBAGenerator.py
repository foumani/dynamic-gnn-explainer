import networkx as nx

from generator.StaticGraphGenerator import StaticGraphGenerator


class StaticBAGenerator(StaticGraphGenerator):
    role_size = 1

    def __init__(self, n_edges: int, *args, **kwargs):
        super(StaticBAGenerator, self).__init__(*args, **kwargs)
        self.n_edges = n_edges
        self.role_size = 1

    def generate(self, node_start: int = 0, role_start: int = 0) -> (nx.Graph, list):
        G = nx.barabasi_albert_graph(n=self.size, m=self.n_edges)
        mapping = {i: i + node_start for i in G.nodes}
        G = nx.relabel_nodes(G, mapping)
        roles = [role_start] * self.size
        return G, roles
