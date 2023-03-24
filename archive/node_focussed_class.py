import numpy as np
import itertools


class Graph:
    def __init__(self, m_0=2):
        self.N = m_0
        self.time = 0
        self.vertices = m_0
        self.degree = np.array([2] * m_0)  # because the graph is complete
        self.__normalisation = 2 * m_0
        self.__probability = self.degree / self.__normalisation

    def iteration(self, N, m):
        while self.vertices < N:
            self.vertices += 1
            new_node = self.vertices
            for edge in range(m+1):
                random_node = choice(self.__probability)
                self.degree[random_node] += 1


class Vertex:
    def __init__(self, id: int, connected_vertices: list):
        self.id = id
        self.degree = 0
        self.connected_vertices = connected_vertices


def choice(probs):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return i
    return -1
