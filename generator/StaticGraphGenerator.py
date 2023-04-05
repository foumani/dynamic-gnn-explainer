import networkx as nx


class StaticGraphGenerator:
    role_size = 0

    def __init__(self, size: int):
        assert size > 0
        self.size = size

    def generate(self, node_start: int = 0, role_start: int = 0) -> (nx.Graph, list): pass
