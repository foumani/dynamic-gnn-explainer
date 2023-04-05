import math
import random

import networkx as nx

from generator.HouseDuoGenerator import HouseDuoGenerator
from generator.StaticBAGenerator import StaticBAGenerator


class DynamicGraphGenerator:
    # TODO: implement a more generic approach for dynamic graph generation
    def __init__(self, T: int, size: int, n_motif: int, p_dynamic_noise: float):
        self.T = T
        self.size = size
        self.n_motif = n_motif
        self.p_noise = p_dynamic_noise

    def generate(self) -> (list, list):
        ba_generator = StaticBAGenerator(size=self.size, n_edges=5)
        house_duo_generator = HouseDuoGenerator()
        G_basis, roles_basis = ba_generator.generate(node_start=0, role_start=0)
        graphs = [G_basis.copy() for _ in range(self.T)]
        roles = roles_basis.copy()
        basis_nodes = set(list(G_basis.nodes()))
        start = self.size
        role_base = 1
        ba_generator = StaticBAGenerator(size=9, n_edges=2)
        for i in range(self.n_motif):
            t_delta, t_motif_start, t_motif_end = 0, 0, 0
            while t_delta == 0:
                t_motif_start, t_motif_end = random.randint(0, self.T - 1), random.randint(0, self.T - 1)
                t_motif_start, t_motif_end = min(t_motif_start, t_motif_end), max(t_motif_start, t_motif_end)
                t_delta = t_motif_end - t_motif_start

            G_start, roles_start = ba_generator.generate(node_start=start, role_start=role_base)
            G_motif_start, roles_motif_start = house_duo_generator.generate(node_start=start, role_start=role_base,
                                                                            house_nodes=[0, 1, 2, 3, 4], super_node=4)
            G_motif_end, roles_motif_end = house_duo_generator.generate(start, role_base, house_nodes=[5, 6, 7, 8, 4],
                                                                        super_node=4)
            G_end, roles_end = ba_generator.generate(node_start=start, role_start=role_base)
            motif = self.morph(G_start, G_motif_start, T=t_motif_start + 1)[0:-1] if t_motif_start > 0 else []
            motif.extend(self.morph(G_motif_start, G_motif_end, T=t_delta + 1)[0:-1])  # We know that t_delta > 0
            motif.extend(
                self.morph(G_motif_end, G_end, T=self.T - t_motif_end) if self.T - t_motif_end > 1 else [G_motif_end])
            graphs = self.join_through(graphs, motif, basis_nodes)
            start += 9
            roles.extend(roles_motif_start)
        return graphs, roles

    def morph(self, G_start: nx.Graph, G_end: nx.Graph, T: int) -> list:
        """
        :param G_start: Graph at the start of sequence.
        :param G_end: Graph at the end of graphs.
        :param T: Length of the time for returned dynamic graph.
        :return: A list of graphs with length t where the first graph is G_start and the last one is G_end.
        """
        force_rem = set(list(G_start.edges())) - set(list(G_end.edges())) - set(
            [(v, u) for u, v in list(G_end.edges())])
        force_add = set(list(G_end.edges())) - set(list(G_start.edges())) - set(
            [(v, u) for u, v in list(G_start.edges())])
        n_noise = max(math.ceil(G_start.number_of_edges() * self.p_noise) - len(force_rem) - len(force_add), 0)
        graphs = [G_start.copy() for _ in range(T)]

        for e in force_add:
            t_add = random.randint(1,
                                   T - 1) if T - 1 > 1 else 1  # We are not going to change G_start, hence start from 1
            for j in range(t_add, T):
                graphs[j].add_edge(*e)

        def connect_through_time(nodes_1: set, nodes_2: set, t_start: int):
            u, v = random.choice(list(nodes_1)), random.choice(list(nodes_2))
            for i in range(t_start, T):
                graphs[i].add_edge(u, v)

        for e in force_rem:
            t_rem = random.randint(1,
                                   T - 1) if T - 1 > 1 else 1  # We are not going to change G_start, hence start from 1
            for j in range(t_rem, T):
                graphs[j].remove_edge(*e)
                components = []
                for c in nx.connected_components(graphs[j]):
                    components.append(c)
                if len(components) > 1:
                    n_noise = max(n_noise - 1, 0)
                    connect_through_time(components[0], components[1], j)

        while n_noise > 0:
            if T - 2 < 1: break # in this case no noise can be added because it
            t_change = random.randint(1, T - 2)  # included start
            t_revert = random.randint(1, T - 1)  # excluded end
            t_change, t_revert = min(t_change, t_revert), max(t_change, t_revert)
            if t_revert - t_change < 1: continue
            n_edges = float(graphs[t_change].number_of_edges())
            n_possible = (graphs[t_change].number_of_nodes() * (graphs[t_change].number_of_nodes() - 1)) / 2
            p_rem = n_edges / n_possible
            if random.random() < p_rem:
                e = random.choice(list(graphs[t_change].edges))
                for t in range(t_change, t_revert):
                    graphs[t].remove_edge(*e)
                    components = nx.connected_components(graphs[t])
                    if len(components) > 1:
                        n_noise = max(n_noise - 1, 0)
                        connect_through_time(components[0], components[1], t)
                n_noise -= 1
            else:
                while True:
                    u = random.choice(list(graphs[t_change].nodes))
                    v = random.choice(list(graphs[t_change].nodes))
                    if graphs[t_change].has_edge(u, v) or u == v: continue
                    for i in range(t_change, t_revert):
                        graphs[i].add_edge(u, v)
                    break
                n_noise -= 2
        return graphs

    @staticmethod
    def join_through(basis: list, struct: list, basis_nodes: set) -> list:
        """
        :param basis_nodes:
        :param struct:
        :param basis:
        :return:
        """
        u, v = random.choice(list(basis_nodes)), random.choice(list(struct[0].nodes))
        graphs = []
        for t in range(len(basis)):
            G = nx.Graph()
            G.add_edges_from(basis[t].edges)
            G.add_edges_from(struct[t].edges)
            G.add_edge(u, v)
            graphs.append(G)
        return graphs
