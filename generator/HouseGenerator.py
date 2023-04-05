import networkx as nx
import numpy as np

from generator.StaticGraphGenerator import StaticGraphGenerator


class HouseGeneratorStatic(StaticGraphGenerator):
    role_size = 5

    def __init__(self):
        super(HouseGeneratorStatic, self).__init__(size=5)

    def generate(self, node_start: int = 0, role_start: int = 0) -> (nx.Graph, list):
        G = nx.Graph()
        G.add_nodes_from([*range(node_start, node_start + 5)])
        G.add_edges_from([
            (node_start, node_start + 1),
            (node_start + 1, node_start + 2),
            (node_start + 2, node_start + 3),
            (node_start + 3, node_start),
            (node_start + 4, node_start),
            (node_start + 4, node_start + 1)
        ])
        roles = np.array([0, 0, 1, 1, 2])
        roles = roles + role_start
        return G, roles.tolist()

