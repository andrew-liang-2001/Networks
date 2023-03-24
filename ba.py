import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from logbin import *
from rustworkx.visualization import mpl_draw
import networkx as nx
import scienceplots
import uncertainties as unc
from collections import Counter
import itertools
import pathlib


plt.style.use(["science"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (6.6942, 4.016538),
                     "axes.labelsize": 15,
                     "legend.fontsize": 12,
                     "xtick.labelsize": 13,
                     "ytick.labelsize": 13,
                     })


#  sys.maxsize is 9223372036854775807


class BarabasiAlbert:
    """Barabasi-Albert model. Nodes are just integers."""

    def __init__(self, N_0: int, graph_type="complete"):
        """For a complete graph, the degree of each node is N_0 - 1"""
        if graph_type not in {"complete"}:
            raise ValueError("Unrecognised graph_type")
        self.G = rx.PyGraph()
        self.N_0 = N_0
        self.m_0 = N_0 * (N_0 - 1) // 2

        self.G.add_nodes_from(range(N_0))
        self.G.extend_from_edge_list(list(itertools.combinations(range(N_0), 2)))
        self._attachment_nodes = [node for node in range(N_0) for _ in range(self.G.degree(node))]

    def nodes(self):
        """
        Return a list of nodes in the graph
        :return: list of nodes
        """
        return self.G.nodes()

    def edges(self):
        return [self.G.get_edge_endpoints_by_index(i) for i in range(self.N_0)]

    def reset(self, m):
        self.__init__(m)

    def degree(self, node):
        """Return the degree of a node"""
        return self.G.degree(node)

    # def drive_3(self, N, m: int) -> None:
    #     """This one is probably broken, but it works???"""
    #     if self.N_0 < m:
    #         raise ValueError("cannot allow multi-graphs")
    #     with tqdm(total=(N - self.N_0)) as pbar:
    #         while self.N_0 < N:
    #             self.G.add_node(self.N_0)
    #             temp_list = []
    #             disallowed_nodes = {}  # faster lookup time, and we don't get duplicates here anyway
    #             for edge in range(m):
    #                 new_node = random.choice(self._attachment_nodes)  # np random choice is O(n) yikes
    #                 while new_node in disallowed_nodes:  # generate another node if it's already been attached.
    #                     new_node = random.choice(self._attachment_nodes)
    #                 temp_list.append(new_node)
    #                 self.G.add_edge(self.N_0, new_node, None)
    #             self._attachment_nodes.extend(temp_list)  # update the attachment nodes
    #             self._attachment_nodes.extend([self.N_0] * m)
    #             self.N_0 += 1
    #             pbar.update(1)

    def drive(self, N: int, m: int, method="PA") -> None:
        """
        Drive the model until N nodes are in the graph. Each iteration adds 1 node and m edges.
        :param method: attachment method.
        :param N: number of nodes in the final graph
        :param m: number of edges added per iteration
        """
        if self.N_0 >= N:
            raise ValueError("Need N > N_0, where N_0 is the number of nodes already in the graph")
        elif method not in {"PA", "RA", "MA"}:
            raise ValueError("Unrecognised method")

        if method == "PA":
            try:
                random.choices(self._attachment_nodes, k=m)
            except TypeError:
                raise ValueError("Cannot use 'PA' after 'RA' method")

            with tqdm(total=(N - self.N_0)) as pbar:
                while self.N_0 < N:
                    self.G.add_node(self.N_0)
                    sampled_nodes = random.choices(self._attachment_nodes, k=m)
                    self.G.extend_from_edge_list([(self.N_0, i) for i in sampled_nodes])
                    self._attachment_nodes.extend(sampled_nodes + [self.N_0] * m)  # update the attachment nodes
                    self.N_0 += 1
                    pbar.update(1)

        elif method == "RA":
            if self._attachment_nodes is None:
                raise ValueError("Cannot use 'RA' after 'PA' to speed up code below")
            self._attachment_nodes = None  # this is not updated so set it to None to prevent accidental use
            with tqdm(total=(N - self.N_0)) as pbar:
                while self.N_0 < N:
                    self.G.add_node(self.N_0)
                    sampled_nodes = random.choices(range(self.N_0), k=m)
                    self.G.extend_from_edge_list([(self.N_0, i) for i in sampled_nodes])
                    self.N_0 += 1
                    pbar.update(1)

        elif method == "MA":  # Mixed attachment method
            if m % 3 != 0:
                raise ValueError("m not divisible by 3")
            r = m // 3
            with tqdm(total=(N - self.N_0)) as pbar:
                while self.N_0 < N:
                    self.G.add_node(self.N_0)
                    sampled_nodes = random.choices(range(self.N_0), k=r)
                    sampled_nodes2 = random.choices(self._attachment_nodes, k=2 * (m-r))

                    for i in range(len(sampled_nodes2), 2):
                        node1, node2 = sampled_nodes2[i], sampled_nodes2[i + 1]
                        while node1 == node2:  # Resample if the nodes are the same to avoid self-loops
                            node2 = random.choice(self._attachment_nodes)

                        self._attachment_nodes.extend([node1, node2])
                        self.G.add_edge(node1, node2)

                    self.G.extend_from_edge_list([(self.N_0, i) for i in sampled_nodes])
                    self._attachment_nodes.extend(sampled_nodes + [self.N_0] * r)  # update the attachment nodes
                    self.N_0 += 1
                    pbar.update(1)

    def node_degrees(self) -> np.ndarray:
        """
        Return an array of the degrees of each node in the graph
        :return:
        """
        # return np.array([self.G.degree(i) for i in range(self.N_0)])
        # list(Counter(self._attachment_nodes).values())
        return np.array([self.G.degree(i) for i in self.G.node_indices()])

    def degree_distribution(self) -> np.ndarray:
        """
        Return the degree distribution of the graph
        :return: degree distribution
        """
        return np.bincount(self.node_degrees())

    def fraction_of_multiedges(self) -> float:
        """
        Return the fraction of multiedges in the graph
        :return: fraction of multiedges
        """
        return 1 - len(self.G.edges()) / self.m_0

    def log_binned_degree_distribution(self, a=1., zeros=False, rm=True) -> (np.ndarray, np.ndarray):
        x, y = logbin(self.node_degrees(), scale=a, zeros=zeros)
        if rm:
            x = x[1:]
            y = y[1:]
        return x, y

    def CDF(self) -> np.ndarray:
        """
        Return the normalised cumulative distribution function (CDF) of the degree distribution
        :return: CDF of the degree distribution
        """
        node_degrees = self.node_degrees()
        return np.cumsum(np.bincount(node_degrees)) / self.N_0

    def CCDF(self) -> np.ndarray:
        """
        Return the complementary cumulative distribution function (CCDF) of the degree distribution
        :return: CCDF of the degree distribution
        """
        return 1 - self.CDF()

    def largest_degree(self) -> int:
        """
        Return the largest degree in the graph
        :return: the largest degree in the graph
        """
        return max(self.node_degrees())

    def draw(self, save_filename=None):
        """
        Draw the graph using matplotlib
        """
        mpl_draw(self.G)
        if save_filename is not None:
            pathlib.Path("graph_drawings").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"graph_drawings/{save_filename}.pdf")
        else:
            plt.show()


def plot_degree_distribution(node_degrees, m, a=1., save_filename=None) -> None:
    """
    Plot a log-log plot of the degree distribution with a power-law fit using matplotlib.pyplot
    :param m: number of edges per iteration in the BA model
    :param node_degrees:
    :param a: log-bin scale
    :param save_filename: filename to save the plot to
    """
    if node_degrees.ndim == 1:
        node_degrees = np.array([node_degrees])
        m = np.array([m])

    for m, node_degrees in zip(m, node_degrees):
        x, y = logbin(node_degrees, scale=a)
        plt.loglog(x, y, ".")

        # noinspection PyTupleAssignmentBalance
        p, pcov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
        y_intercept = unc.ufloat(p[1], np.sqrt(pcov[1][1]))
        gradient = unc.ufloat(p[0], np.sqrt(pcov[0][0]))
        plt.loglog(x, np.exp(p[1]) * x ** p[0], "--", label=f"slope = {gradient:.P}")
        x_arr = np.linspace(1, max(x), 100_000)
        plt.loglog(x_arr, p_pa_analytical(x_arr, m), "--", label="Analytical")

    plt.xlabel("Degree k")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(f"{save_filename}.pdf")


def p_pa_analytical(k, m):
    """
    Return the probability of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: probability of a node with degree k
    """
    return 2 * m * (m + 1) / (k * (k + 1) * (k + 2))


def p_ra_analytical(k, m):
    """
    Return the probability of a node with degree k with random attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: probability of a node with degree k
    """
    return (1 / (m+1)) * (m / (m + 1)) ** (k - m)


def p_ma_analytical(k, m):
    pass


def CDF_pa_analytical(k, m):
    """
    Return the CDF of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: CDF of a node with degree k
    """
    return 1 - (m ** 2 + m) / ((k+1) * (k+2))


def CCDF_pa_analytical(k, m):
    """
    Return the CDF of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: CDF of a node with degree k
    """
    return (m ** 2 + m) / ((k+1) * (k+2))


def CCDF_powerlaw(x_min, x, alpha):
    """
    Return the CCDF of a power law distribution
    :param k:
    :param m:
    :return:
    """
    return (x/x_min)**(1-alpha)


def CDF_ra_analytical(k, m):
    """
    Return the CDF of a node with degree k with random attachment
    :param k:
    :param m:
    :return:
    """
    pass


def CDF_powerlaw(k, m):
    """
    Return the CDF of a power law distribution
    :param k:
    :param m:
    :return:
    """
    pass


def k1_pa_analytical(N, m):
    """
    Return the analytical value of k_1, the largest degree
    :param m: number of edges per iteration in the BA model
    :param N: number of nodes in the graph
    :return: analytical value of k1
    """
    return -0.5 + np.sqrt((1 + 4*N*m*(m+1)))/2


def k_1_ra_analytical(N, m):
    """
    Return the analytical value of k_1, the largest degree
    :param m: number of edges per iteration in the RA model
    :param N: number of nodes in the graph
    :return: analytical value of k1
    """
    return m - (np.log(N) / (np.log(m) - np.log(m+1)))


def plot_CDF(m):
    k = np.arange(m, m+20, 1)
    plt.plot(k, CDF_pa_analytical(k, m))
    plt.xlabel("Degree k")
    plt.ylabel("$C(k)$")
    plt.show()


# def power_law(x, a, b):
#     return a * x ** b
#
# def pad_logbin(xy_list):
#


def convert_rustworkx_to_networkx(graph):
    """Convert a rustworkx PyGraph or PyDiGraph to a networkx graph.
    See https://qiskit.org/documentation/retworkx/dev/networkx.html"""
    edge_list = [(
        graph[x[0]], graph[x[1]],
        {'weight': x[2]}) for x in graph.weighted_edge_list()]

    if isinstance(graph, rx.PyGraph):
        if graph.multigraph:
            return nx.MultiGraph(edge_list)
        else:
            return nx.Graph(edge_list)
    else:
        if graph.multigraph:
            return nx.MultiDiGraph(edge_list)
        else:
            return nx.DiGraph(edge_list)


# if __name__ == '__main__':
#     # ba = BarabasiAlbert(4)
#     # ba.draw("complete graph")
#     # plot_degree_distribution(ba.node_degrees(), m=3, a=1.2)
#
#     # ba.draw()
