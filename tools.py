"""Contains helper functions for generating data and plotting results"""

from dataclasses import dataclass
from analytical_functions import *
import matplotlib.pyplot as plt
from ba import BarabasiAlbert
import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (6.6942, 4.016538),
                     "axes.labelsize": 15,
                     "legend.fontsize": 11,
                     "xtick.labelsize": 13,
                     "ytick.labelsize": 13,
                     })


@dataclass
class Parameters:
    """Result of running a random graph"""
    m: int
    N: np.ndarray
    method: str
    iterations: int


def pad_logbin(xy_list):
    """
    Pad the logbin arrays, so they are all the same length
    :param xy_list:
    :return: padded xy array
    """
    # TODO: reshape the final array
    x_arr = np.array([x for x, _ in xy_list])  # get the x arrays
    x_arr = x_arr[np.argmax([len(x) for x in x_arr])]  # find the x which has the most bins

    for i, x in enumerate(xy_list):
        xy_list[i][0] = x_arr
        xy_list[i][1] = np.pad(xy_list[i][1], (0, len(x_arr) - len(xy_list[i][1])), "constant")  # not ragged

    return np.array(xy_list)


def generate_k1_data(N_arr, method="PA", m=2, iterations=1000, a=1.1):
    """
    Helper function to generate data for the largest degree, relevant to both the largest degree plot and data collapse
    :param iterations:
    :param m:
    :param a:
    :param method:
    :param N_arr:
    :return:
    """

    result_k1 = []
    result_k1_std_dev = []
    result_data_collapse = []
    result_data_collapse_std_dev = []

    for N in N_arr:  # varied N
        temp = []
        temp2 = []
        for _ in range(iterations):  # number of repeats
            ba = BarabasiAlbert(m + 1)
            ba.drive(N, m, method=method)
            temp.append(ba.largest_degree())
            temp2.append(list(ba.log_binned_degree_distribution(a=a, zeros=False, rm=True)))

        result_k1.append(np.mean(temp))
        result_k1_std_dev.append(np.std(temp))

        temp2 = pad_logbin(temp2)
        result_data_collapse.append(np.mean(temp2, axis=0))
        result_data_collapse_std_dev.append(np.std(temp2, axis=0))

    result_k1 = np.array(result_k1)
    result_k1_std_dev = np.array(result_k1_std_dev)

    return result_k1, result_k1_std_dev, result_data_collapse, result_data_collapse_std_dev


def generate_pk_data(method, N: int = 1_000_000, m_arr=None, a=1.2) -> tuple[list, list]:
    """
    Generate log-binned p(k) distribution for a given method and fixed N
    :param method:
    :param N: Number of final nodes
    :param m_arr: array of m values
    :param a: log-bin scaling
    :return:
    """
    if m_arr is None:
        m_arr = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    result_mean = []
    result_std_dev = []

    for m in m_arr:
        temp = []
        for _ in range(50):
            ba = BarabasiAlbert(m + 1)
            ba.drive(N, m, method=method)
            temp.append(list(ba.log_binned_degree_distribution(a=a, zeros=True, rm=True)))

        temp = pad_logbin(temp)

        result_mean.append(np.mean(temp, axis=0))
        result_std_dev.append(np.std(temp, axis=0))

    return result_mean, result_std_dev


def plot_k1(method, result, result_std_dev, N_arr=None, m=2, repeats=100) -> None:
    if N_arr is None:
        N_arr = np.array([100, 300, 1000, 3000, 10_000, 30_000, 100_000])

    plt.errorbar(N_arr, result, fmt=".", yerr=result_std_dev / np.sqrt(repeats), label="Empirical")
    N_dense = np.arange(1, max(N_arr), 1)

    if method == "PA":
        k1_theory = k1_pa_analytical(m, N_dense)
    elif method == "RA":
        k1_theory = k1_ra_analytical(m, N_dense)
    elif method == "MA":
        raise NotImplementedError
    else:
        raise ValueError("Method not recognised")

    plt.plot(N_dense, k1_theory, "--", label="Theoretical")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("$k_1$")
    plt.legend()
    plt.tight_layout()


def plot_data_collapse(method, result, result_std_dev, k1_result, N_arr=None, m=2, repeats=1000) -> None:
    if N_arr is None:
        N_arr = np.array([100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6942, 4.016538), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    for i in range(len(result)):
        k = result[i][0]
        k_1_empirical = k1_result[i]
        if method == "PA":
            p_theory = p_pa_analytical(k, m)
        elif method == "RA":
            p_theory = p_ra_analytical(k, m)
        elif method == "MA":
            p_theory = p_ma_analytical(k, m)
        else:
            raise ValueError("Method not recognised")

        ax0.errorbar(k / k_1_empirical, result[i][1] / p_theory,
                     fmt=".", label=f"N = {N_arr[i]}")

        ax1.errorbar(k / k_1_empirical, np.zeros(len(k)),
                     yerr=result_std_dev[i][1] / (p_theory * np.sqrt(repeats)),
                     fmt=".", label=f"N = {N_arr[i]}", ms=0, capsize=2)

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$k/k^{em}_1$")
    ax0.set_ylabel(r"$p(k)/p^{th}\left( k \right)$")
    # ax1.set_ylabel(r"$\sigma_{\ln{(p(k))}} / p^{th}\left( k \right)$")
    ax0.legend()
    plt.tight_layout()
