import numpy as np
import itertools


class Graph:
    def __init__(self, m_0=2):
        self.N = m_0
        self.time = 0
        self.edges = list(itertools.combinations(range(m_0), 2))
        self.vertices = m_0
        self.degree = [2] * m_0  # because the graph is complete
        self.normalisation = len(self.edges)

    def iteration(self, N, m):
        while self.vertices < N:
            self.vertices += 1
            new_node = self.vertices
            for edge in range(m+1):
                self.edges.append((new_node, edge))


class Edge:
    def __init__(self, link: tuple):
        self.link = link


class Vertex:
    def __init__(self, id: int):
        self.id = id
        self.degree = 0


def choice(options, probs):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]
