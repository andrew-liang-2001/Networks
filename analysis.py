import dill as pickle
from ba import *
import scipy.stats
import functools
import powerlaw
import uncertainties as unc

N = 1_000_000
m_arr = np.array([2, 4, 8, 16, 32, 64, 128])
p = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']  # plot colours

# %%
ba = BarabasiAlbert(102)
ba.drive(1_000_000, 100)
plot_degree_distribution(np.array(ba.node_degrees2()), m=100, a=1.2)
plt.show()

# %%
ba = BarabasiAlbert(3)
for node in ba.nodes():
    print(node)

ba.degree(0)
ba.edges()

# %% 1.3.2 unaveraged
result132 = []
result132_std_dev = []

for m in m_arr:
    ba = BarabasiAlbert(m+1)
    ba.drive(N, m)
    result132.append(ba.log_binned_degree_distribution(a=1.2, zeros=True))

for data in result132:
    plt.loglog(data[0], data[1], label=f"m = {m}")

# %%
ba = BarabasiAlbert(3)
ba.drive(100_000, 2)
result = powerlaw.Fit(ba.node_degrees())
print(result.alpha, result.xmin, result.sigma)
result.plot_pdf(linestyle="", label="Powerlaw", marker="o", ms=1.8)
result.plot_ccdf(linestyle="", label="Powerlaw", marker="o", ms=1.8)
plt.show()

#%%
ba = BarabasiAlbert(3)
ba.drive(10_000, 2)
# noinspection PyTypeChecker
result = powerlaw.Fit(ba.node_degrees())
print(result.alpha, result.xmin, result.sigma)

# %% using powerlaw package https://pypi.org/project/powerlaw/
N = 1_000_000
# create array of length m_arr
powerlaw_arr = np.zeros(len(m_arr))
powerlaw_std_dev = np.zeros(len(m_arr))
powerlaw_xmin = np.zeros(len(m_arr))

for i, m in enumerate(m_arr):
    ba = BarabasiAlbert(m+1)
    ba.drive(N, m)
    # noinspection PyTypeChecker
    result = powerlaw.Fit(ba.node_degrees())
    powerlaw_arr[i] = result.alpha
    powerlaw_std_dev[i] = result.sigma
    powerlaw_xmin[i] = result.xmin

# %% save 1.3.2 powerlaw
np.savez("data/powerlaw_1mil.npz", powerlaw_arr, powerlaw_std_dev)

# %% 1.3.2 load powerlaw
powerlaw_arr, powerlaw_std_dev = np.load("data/powerlaw_1mil.npz")

# %% 1.3.2
result132_mean = []
result132_std_dev = []
result132_SE = []
repeats132 = []
N = 1_000_000

for j, m in enumerate(m_arr):
    temp = []
    for _ in range(100):
        ba = BarabasiAlbert(m+1)
        ba.drive(N, m)
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=False)))

    x_arr = np.array([x for x, _ in temp])  # get the x arrays
    x_arr = x_arr[np.argmax([len(x) for x in x_arr])]  # find the x which has the most bins

    repeats132_m = np.zeros_like(x_arr, dtype=int)  # Initialize the repeats132 for the current m

    for i, x in enumerate(temp):
        temp[i][0] = x_arr
        original_frequencies = temp[i][1]
        # pad the y arrays with zeros up to the length of the longest x array
        temp[i][1] = np.pad(temp[i][1], (0, len(x_arr) - len(temp[i][1])), "constant")  # array no longer ragged
        repeats132_m[:len(original_frequencies)] += (original_frequencies != 0)

    temp = np.array(temp)

    repeats132.append(repeats132_m)  # Add the repeats132_m to the main repeats132 array

    result132_mean.append(np.mean(temp, axis=0))
    result132_std_dev.append(np.std(temp, axis=0))

for std, repeats in zip(result132_std_dev, repeats132):
    result132_SE.append(np.divide(std[1], np.sqrt(repeats)))

# %% 1.3.2 save
with open("data/result132.pkl", "wb") as f:
    pickle.dump((result132_mean, result132_std_dev, repeats132, result132_SE), f)

# %% 1.3.2 load
with open("data/result132.pkl", "rb") as f:
    result132_mean, result132_std_dev, repeats132, result132_SE = pickle.load(f)

# %% CHECK
for i, (mean, std_dev, SE) in enumerate(zip(result132_mean, result132_std_dev, result132_SE)):
    print("std_dev", std_dev[1])
    print("SE", SE)

"""NOTE: the result_std_dev[0] is just the bins. So it's redundant"""
# %% 1.3.2 plot
for i, (mean, std_dev, SE) in enumerate(zip(result132_mean, result132_std_dev, result132_SE)):
    # propagate the standard deviations through the log using first order Taylor expansion
    plt.errorbar(mean[0], mean[1], yerr=std_dev[1]/np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i], ms=3)
    k_dense = np.linspace(1, max(mean[0]), 1000)
    plt.plot(k_dense, p_pa_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/132.pdf", format="pdf")
plt.show()

# %%
ba = BarabasiAlbert(10)
ba.drive(1_000_000, 3)

# %%
# plot degree distribution
plot_degree_distribution(ba.node_degrees(), m=3)

# %%  CCDF plot
plt.loglog(ba.CCDF(), ".", label="Empirical")
k = np.arange(0, max(ba.node_degrees())+1, 1)
plt.loglog(k, CCDF_pa_analytical(k, m=3), "--", label="Analytical")
# plt.xlim(1, 20)
# plt.ylim(0, 1)
plt.xlabel("k")
plt.ylabel("$C(k)$")
plt.legend()
plt.show()
# ba.cumulative_node_degrees()

# %%
PDF_scipy = functools.partial(p_pa_analytical, m=3)
# scipy.stats.ks_1samp(ba.node_degrees(), CDF_pa_analytical, mode="exact", args=(3,))
scipy.stats.ks_1samp(ba.node_degrees(), ba.CDF, mode="exact")

# %% 1.4.2
N_arr = [10, 100, 1000, 10000, 100000, 1000000]

result142 = []
result142_std_dev = []
result143 = []

for N in N_arr:  # varied N
    temp = []
    temp2 = []
    for _ in range(10):  # number of repeats
        ba = BarabasiAlbert(3)
        ba.drive(N, 2)
        temp.append(ba.largest_degree())
        temp2.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=False)))
    result142.append(np.mean(temp))
    result142_std_dev.append(np.std(temp))
    result143.append(np.mean(temp2, axis=0))

result142 = np.array(result142)
result142_std_dev = np.array(result142_std_dev)
result143 = np.array(result143)

# %% 1.4.2 save
np.savez("data/142_10repeats_m=3.npz", N_arr, result142, result142_std_dev)

# %% 1.4.2 load
N_arr, result142, result142_std_dev = np.load("data/142_10repeats_m=3.npz")

# %% 1.4.2 plot
plt.errorbar(N_arr, result142, fmt=".", yerr=result142_std_dev, label="Empirical")
N_dense = np.arange(1, 10_000_000, 1)
plt.plot(N_dense, k1_pa_analytical(m=2, N=N_dense), "--", label="Analytical")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("$k_1$")
plt.legend()
plt.tight_layout()
plt.show()

# %% 1.4.3


for i, (mean, std_dev, k_1) in enumerate(zip(result132_mean, result142_std_dev, result142)):
    # propagate the standard deviations through the log using first order Taylor expansion
    plt.errorbar(mean[0]/result142, mean[1], yerr=std_dev[1]/np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i], ms=3)
    k_dense = np.linspace(1, max(mean[0]), 1000)
    plt.plot(k_dense, p_pa_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/132.pdf", format="pdf")
plt.show()

# %% 2.2
result22_mean = []
result22_std_dev = []
result22_SE = []
repeats22 = []
N = 1_000_000

for j, m in enumerate(m_arr):
    temp = []
    for _ in range(50):
        ba = BarabasiAlbert(m+1)
        ba.drive(N, m, "RA")
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=True)))

    x_arr = np.array([x for x, _ in temp])  # get the x arrays
    x_arr = x_arr[np.argmax([len(x) for x in x_arr])]  # find the x which has the most bins

    repeats22_m = np.zeros_like(x_arr, dtype=int)  # Initialize the repeats22 for the current m

    for i, x in enumerate(temp):
        temp[i][0] = x_arr
        original_frequencies = temp[i][1]
        # pad the y arrays with zeros up to the length of the longest x array
        temp[i][1] = np.pad(temp[i][1], (0, len(x_arr) - len(temp[i][1])), "constant")  # array no longer ragged
        repeats22_m[:len(original_frequencies)] += (original_frequencies != 0)

    temp = np.array(temp)

    repeats22.append(repeats22_m)  # Add the repeats22_m to the main repeats22 array

    result22_mean.append(np.mean(temp, axis=0))
    result22_std_dev.append(np.std(temp, axis=0))

for std, repeats in zip(result22_std_dev, repeats22):
    result22_SE.append(np.divide(std[1], np.sqrt(repeats)))

# %% 2.2 save
with open("data/result22.pkl", "wb") as f:
    pickle.dump((result22_mean, result22_std_dev, result22_SE, repeats22), f)

# %% 2.2 load
with open("data/result22.pkl", "rb") as f:
    result22_mean, result22_std_dev, result22_SE, repeats22 = pickle.load(f)

# %% 2.2 plot
for i, (mean, std_dev, SE) in enumerate(zip(result22_mean, result22_std_dev, result22_SE)):
    # propagate the standard deviations through the log using first order Taylor expansion
    plt.errorbar(mean[0], mean[1], yerr=std_dev[1]/np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i], ms=3)
    k_dense = np.linspace(1, max(mean[0]), 1000)
    plt.plot(k_dense, p_ra_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()
plt.show()
plt.tight_layout()
# plt.savefig("plots/22.pdf", format="pdf")

# %% 2.3

# %% 3.2
result32_mean = []
result32_std_dev = []
result32_SE = []
repeats32 = []
# N = 1_000_000
N = 100_000
m32_arr = np.array([3, 9, 27, 81, 243])

for j, m in enumerate(m32_arr):
    temp = []
    for _ in range(10):
        ba = BarabasiAlbert(m+1)
        ba.drive(N, m, "MA")
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=True)))

    x_arr = np.array([x for x, _ in temp])  # get the x arrays
    x_arr = x_arr[np.argmax([len(x) for x in x_arr])]  # find the x which has the most bins

    repeats32_m = np.zeros_like(x_arr, dtype=int)  # Initialize the repeats32 for the current m

    for i, x in enumerate(temp):
        temp[i][0] = x_arr
        original_frequencies = temp[i][1]
        # pad the y arrays with zeros up to the length of the longest x array
        temp[i][1] = np.pad(temp[i][1], (0, len(x_arr) - len(temp[i][1])), "constant")  # array no longer ragged
        repeats32_m[:len(original_frequencies)] += (original_frequencies != 0)

    temp = np.array(temp)

    repeats32.append(repeats32_m)  # Add the repeats32_m to the main repeats32 array

    result32_mean.append(np.mean(temp, axis=0))
    result32_std_dev.append(np.std(temp, axis=0))

for std, repeats in zip(result32_std_dev, repeats32):
    result32_SE.append(np.divide(std[1], np.sqrt(repeats)))

# %% save
with open("data/result32.pkl", "wb") as f:
    pickle.dump((result32_mean, result32_std_dev, result32_SE, repeats32), f)

# %% load
with open("data/result32.pkl", "rb") as f:
    result32_mean, result22_std_dev, result32_SE, repeats32 = pickle.load(f)
# %%
for i, (mean, std_dev, SE) in enumerate(zip(result32_mean, result32_std_dev, result32_SE)):
    # propagate the standard deviations through the log using first order Taylor expansion
    # std_dev[1] = std_dev[1] / mean[1]
    plt.errorbar(mean[0], mean[1], yerr=std_dev/np.sqrt(50), fmt=".", label=f"m = {m32_arr[i]}", color=p[i], ms=3)
    # k_dense = np.linspace(1, max(mean[0]), 1000)
    # plt.plot(k_dense, p_pa_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig("plots/132.pdf", format="pdf")

# %%
