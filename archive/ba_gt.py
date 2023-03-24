import numpy as np
import rustworkx as rx
import graph_tool as gt
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from logbin import *
import networkx as nx
import scienceplots
import uncertainties as unc
from collections import Counter
from line_profiler_pycharm import profile
import itertools
import numba as nb
import pathlib


plt.style.use(["science"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (6.6942, 4.016538),
                     "axes.labelsize": 15,
                     "legend.fontsize": 12,
                     "xtick.labelsize": 13,
                     "ytick.labelsize": 13,
                     'axes.prop_cycle': plt.cycler(color=plt.cm.tab10.colors)})


#  sys.maxsize is 9223372036854775807

class BarabasiAlbert:
    """Barabasi-Albert model. Nodes are just integers."""

    def __init__(self, N_0: int, graph_type="complete"):
        if graph_type not in {"complete"}:
            raise ValueError("Unrecognised graph_type")
        self.G = gt.Graph(directed=False)
        self.N_0 = N_0
        m_0 = N_0 * (N_0 - 1) // 2
        self.G.add_edge_list(list(itertools.combinations(range(N_0), 2)))
        self._attachment_nodes = [node.out_degree() for node in self.G.vertices()]

    def reset(self, m):
        self.__init__(m)

    def degree(self, node):
        """Return the degree of a node"""
        return self.G.degree(node)

    def drive_3(self, N, m: int) -> None:
        """This one is probably broken, but it works???"""
        if self.N_0 < m:
            raise ValueError("cannot allow multi-graphs")
        with tqdm(total=(N - self.N_0)) as pbar:
            while self.N_0 < N:
                self.G.add_node(self.N_0)
                temp_list = []
                disallowed_nodes = {}  # faster lookup time, and we don't get duplicates here anyway
                for edge in range(m):
                    new_node = random.choice(self._attachment_nodes)  # np random choice is O(n) yikes
                    while new_node in disallowed_nodes:  # generate another node if it's already been attached.
                        new_node = random.choice(self._attachment_nodes)
                    temp_list.append(new_node)
                    self.G.add_edge(self.N_0, new_node, None)
                self._attachment_nodes.extend(temp_list)  # update the attachment nodes
                self._attachment_nodes.extend([self.N_0] * m)
                self.N_0 += 1
                pbar.update(1)

    def drive(self, N, m: int):
        """multi-graph allowed"""
        with tqdm(total=(N - self.N_0)) as pbar:
            while self.N_0 < N:
                new_nodes = random.choices(self._attachment_nodes, k=m)
                self.G.add_edge_list([(self.N_0, i) for i in new_nodes])
                self._attachment_nodes.extend(new_nodes + [self.N_0] * m)  # update the attachment nodes
                self.N_0 += 1
                pbar.update(1)

    def degree_distribution(self):
        return np.array([node.out_degree() for node in self.G.vertices()])


def p_analytical(k, m):
    """
    Return the probability of a node with degree k
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: probability of a node with degree k
    """
    return 2 * m * (m + 1) / (k * (k + 1) * (k + 2))


if __name__ == '__main__':
    ba = BarabasiAlbert(3)
    ba.drive(1_000_000, 5)
    plot_degree_distribution(ba.degree_distribution(), m=3, a=1.2)


    # ba.plot_degree_distribution(a=1.2)
    # ba.draw()
