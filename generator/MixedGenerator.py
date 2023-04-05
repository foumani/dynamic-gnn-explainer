import networkx as nx
import numpy as np

from generator.StaticGraphGenerator import StaticGraphGenerator


class MixedGenerator:
    role_size = 0

    def __init__(self,
                 basis_generator: StaticGraphGenerator,
                 motif_generator: StaticGraphGenerator):
        self.basis_generator = basis_generator
        self.motif_generator = motif_generator

    def generate(self, node_start: int = 0, role_start: int = 0, n_motif: int = 1) -> (nx.Graph, list):
        G, roles = self.basis_generator.generate(node_start=node_start, role_start=role_start)
        plugins = (np.random.choice(self.basis_generator.size, n_motif, replace=True) + node_start).tolist()
        node_start += G.number_of_nodes()
        role_start += self.basis_generator.role_size
        for g in range(n_motif):
            G_motif, roles_motif = self.motif_generator.generate(node_start=node_start, role_start=role_start)
            G.add_nodes_from(list(G_motif.nodes))
            G.add_edges_from(list(G_motif.edges))
            G.add_edge(plugins[g], node_start)
            roles.extend(roles_motif)
            node_start += self.motif_generator.size
        return G, roles
