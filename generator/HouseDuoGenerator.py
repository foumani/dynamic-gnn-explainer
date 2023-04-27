import random

import networkx as nx

from generator.HouseGenerator import HouseGeneratorStatic
from generator.StaticBAGenerator import StaticBAGenerator
from generator.StaticGraphGenerator import StaticGraphGenerator


class HouseDuoGenerator(StaticGraphGenerator):
    role_size = 2

    def __init__(self):
        super(HouseDuoGenerator, self).__init__(size=9)

    def generate(self,
                 node_start: int = 0,
                 role_start: int = 0,
                 house_nodes: list = None,
                 super_node: int = 4) -> (nx.Graph, list):
        house_nodes = house_nodes if house_nodes else [0, 1, 2, 3, 4]
        # creating roles
        roles = [role_start] * 9
        roles[house_nodes[super_node]] = role_start + 1

        house_nodes = [i + node_start for i in house_nodes]
        assert len(house_nodes) == 5
        other_nodes = list(set([i for i in range(node_start, node_start + 9)]) - set(house_nodes))
        house_generator = HouseGeneratorStatic()
        ba_generator = StaticBAGenerator(n_edges=2, size=4)
        house, _ = house_generator.generate(node_start=node_start)
        mapping_graph = {list(house.nodes)[i]: house_nodes[i] for i in range(5)}
        house = nx.relabel_nodes(house, mapping_graph)
        other, _ = ba_generator.generate(node_start=node_start + 5)
        mapping_graph = {list(other.nodes)[i]: other_nodes[i] for i in range(4)}
        other = nx.relabel_nodes(other, mapping_graph)

        G = nx.Graph()
        G.add_edges_from(house.edges)
        G.add_edges_from(other.edges)
        u, v = random.choice(list(house.nodes)), random.choice(list(other.nodes))
        G.add_edge(u, v)
        # roles = [role_start] * 9
        # roles[house_nodes[super_node]] = role_start + 1
        return G, roles
