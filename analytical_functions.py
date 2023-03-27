"""This file stores all the analytical functions, such as the PDF and ECDF"""

import numpy as np
import scipy


def p_pa_analytical(k, m):
    """
    Return the probability of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration
    :return: probability of a node with degree k
    """
    return 2 * m * (m + 1) / (k * (k + 1) * (k + 2))


def p_ra_analytical(k, m):
    """
    Return the probability of a node with degree k with random attachment
    :param k: degree of node
    :param m: number of edges per iteration
    :return: probability of a node with degree k
    """
    return (1 / (m + 1)) * (m / (m + 1)) ** (k - m)


def p_ma_analytical(k, m):
    """
    Return the probability of a node with degree k with mixed attachment
    :param k: degree of node
    :param m: number of edges per iteration
    :return: probability of a node with degree k
    """
    return (9 / (9 + 5 * m)) * np.exp(scipy.special.gammaln(5 * m / 6 + 5 / 2) - scipy.special.gammaln(5 * m / 6)) * \
        np.exp(scipy.special.gammaln(k + m / 2) - scipy.special.gammaln(k + m / 2 + 5 / 2))


def CDF_pa_analytical(k, m):
    """
    Return the ECDF of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: ECDF of a node with degree k
    """
    return 1 - (m ** 2 + m) / ((k + 1) * (k + 2))


def CCDF_pa_analytical(k, m):
    """
    Return the ECDF of a node with degree k with preferential attachment
    :param k: degree of node
    :param m: number of edges per iteration in the BA model
    :return: ECDF of a node with degree k
    """
    return (m ** 2 + m) / ((k + 1) * (k + 2))


def CCDF_powerlaw(x_min, x, alpha):
    """
    Return the CCDF of a power law distribution
    :param k:
    :param m:
    :return:
    """
    return (x / x_min) ** (1 - alpha)


def CDF_ra_analytical(k, m):
    """
    Return the ECDF of a node with degree k with random attachment
    :param k:
    :param m:
    :return:
    """
    pass


def CDF_powerlaw(k, m):
    """
    Return the ECDF of a power law distribution
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
    return -0.5 + np.sqrt((1 + 4 * N * m * (m + 1))) / 2


def k1_ra_analytical(N, m):
    """
    Return the analytical value of k_1, the largest degree
    :param m: number of edges per iteration in the RA model
    :param N: number of nodes in the graph
    :return: analytical value of k1
    """
    return m - (np.log(N) / (np.log(m) - np.log(m + 1)))


def D_threshold(alpha, m, n):
    """
    Return the Kolmogorov-Smirnov distance threshold for a given significance level
    :param alpha:
    :param m:
    :return:
    """
    return np.sqrt(-np.log(alpha/2)*(1+(m/n))/(2*m))
