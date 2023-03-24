import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from logbin import *
import scienceplots
import uncertainties as unc
from collections import Counter
from line_profiler_pycharm import profile

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
    def __init__(self, m_0):
        # make a graph of n nodes and m_0 edges
        self.G = nx.complete_graph(m_0)
        self.N = m_0
        self.time = 0
        self.attachment_nodes = [node for node in range(m_0) for _ in range(m_0)]

    def reset(self, m):
        self.__init__(m)

    def degree(self, node):
        """Return the degree of a node"""
        return self.G.degree(node)

    def drive_1(self, N_max, m):
        with tqdm(total=(N_max - self.N)) as pbar:
            while self.N < N_max:
                self.G.add_node(self.N)
                new_node = self.N
                for edge in range(m + 1):
                    preferred_node = np.random.choice(self.attachment_nodes)  # np random choice is O(n) yikes
                    if preferred_node == new_node:
                        preferred_node = np.random.choice(self.attachment_nodes)
                    self.attachment_nodes.append(preferred_node)
                    self.G.add_edge(new_node, preferred_node)
                self.attachment_nodes.extend([new_node] * m)
                self.N += 1
                self.time += 1
                pbar.update(1)

    @profile
    def drive_3(self, N_max, m: int):
        """This one is probably broken, but it works???"""
        with tqdm(total=(N_max - self.N)) as pbar:
            while self.N < N_max:
                self.G.add_node(self.N)
                s = self.N
                temp_list = []
                disallowed_nodes = {}  # faster lookup time, and we don't get duplicates here anyway
                for edge in range(m):
                    new_node = random.choice(self.attachment_nodes)  # np random choice is O(n) yikes
                    while new_node in disallowed_nodes:  # generate another node if it's already been attached.
                        new_node = random.choice(self.attachment_nodes)
                    temp_list.append(new_node)
                    self.G.add_edge(s, new_node)
                self.attachment_nodes.extend(temp_list)  # update the attachment nodes
                self.attachment_nodes.extend([s] * m)
                self.N += 1
                self.time += 1
                pbar.update(1)

    def drive_2(self, N_max, m=1):
        with tqdm(total=(N_max - self.N)) as pbar:
            while self.N < N_max:
                self.G.add_node(self.N)
                s = self.N
                for edge in range(m):
                    # choose random edge from graph
                    while True:
                        random_edge = random.choice(list(self.G.edges))  # casting list sucks
                        new_node = random.choice(random_edge)
                        if new_node != s:
                            break
                    self.G.add_edge(s, new_node)
                self.time += 1
                self.N += 1
                pbar.update(1)

    def visualise(self):
        nx.draw(self.G)
        plt.show()

    def degree_distribution(self):
        """Return the degree distribution of the graph"""
        return nx.degree_histogram(self.G)

    def get_degree(self):
        return [self.G.degree(node) for node in self.G.nodes]

    # def get_degree(self):
    #     return [self.G.degree(i) for i in range(self.N_0)]

    # def average_degree(self):
    #     """Return the average degree of the graph"""
    #     return np.mean(self.node_degrees())


if __name__ == '__main__':
    ba = BarabasiAlbert(2)
    ba.drive_3(100_000, 2)
    # # print(ba.attachment_nodes)
    # # ba.draw()
