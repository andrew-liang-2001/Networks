"""All analysis code for the project. Use pickle when arrays are ragged"""

import dill as pickle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import powerlaw
from tools import *
import uncertainties as unc
from analytical_functions import *

N = 1_000_000
m_arr = np.array([2, 4, 8, 16, 32, 64, 128])
N_arr = np.array([100, 300, 1000, 3000, 10_000, 30_000, 100_000])
p = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']  # plot colours


# %% using powerlaw package https://pypi.org/project/powerlaw/
powerlaw_arr = np.zeros(len(m_arr))
powerlaw_std_dev = np.zeros(len(m_arr))
powerlaw_xmin = np.zeros(len(m_arr))

for i, m in enumerate(m_arr):
    ba = BarabasiAlbert(m + 1)
    ba.drive(N, m)
    # noinspection PyTypeChecker
    result = powerlaw.Fit(ba.node_degrees())
    powerlaw_arr[i] = result.alpha
    powerlaw_std_dev[i] = result.sigma
    powerlaw_xmin[i] = result.xmin

# %%
print(powerlaw_arr, powerlaw_std_dev, powerlaw_xmin)

# %% save 1.3.2 powerlaw
with open("data/powerlaw_1mil.pkl", "wb") as f:
    pickle.dump((powerlaw_arr, powerlaw_std_dev, powerlaw_xmin), f)

# %% 1.3.2 load powerlaw
with open("data/powerlaw_1mil.pkl", "rb") as f:
    powerlaw_arr, powerlaw_std_dev, powerlaw_xmin = pickle.load(f)

# %% 1.3.2
result132_mean = []
result132_std_dev = []
result132_SE = []
N = 1_000_000

for m in m_arr:
    temp = []
    for _ in range(2):
        ba = BarabasiAlbert(m + 1)
        ba.drive(N, m)
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=False)))

    temp = pad_logbin(temp)

    result132_mean.append(np.mean(temp, axis=0))
    result132_std_dev.append(np.std(temp, axis=0))

# %% 1.3.2 save
with open("data/result132.pkl", "wb") as f:
    pickle.dump((result132_mean, result132_std_dev, result132_SE), f)

# %% 1.3.2 load
with open("data/result132.pkl", "rb") as f:
    result132_mean, result132_std_dev, repeats132, result132_SE = pickle.load(f)

# %% 1.3.2 plot
fig, ax = plt.subplots()

for i, (mean, std_dev, SE) in enumerate(zip(result132_mean, result132_std_dev, result132_SE)):
    plt.errorbar(mean[0], mean[1], yerr=std_dev[1] / np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i], ms=3)
    k_dense = np.linspace(1, max(mean[0]), 100_000)
    plt.plot(k_dense, p_pa_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()

ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])

# plot the data in the inset axis
for i, (mean, std_dev, SE) in enumerate(zip(result132_mean, result132_std_dev, result132_SE)):
    ax_inset.errorbar(mean[0], mean[1], yerr=std_dev[1] / np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i],
                      ms=3)
    k_dense = np.linspace(1, max(mean[0]), 100_000)
    ax_inset.plot(k_dense, p_pa_analytical(k_dense, m=m_arr[i]), "--", color=p[i], linewidth=0.5)

ax_inset.tick_params(axis='both', labelsize=8)
ax_inset.set_xlim([1e3, 2e3])
ax_inset.set_ylim([3e-9, 1e-8])

ax_inset.set_xticklabels([], visible=False)
ax_inset.set_yticklabels([], visible=False)
ax_inset.set_xscale("log")
ax_inset.set_yscale("log")
# plt.minorticks_off()

mark_inset(ax, ax_inset, loc1=1, loc2=2, fc="none", ec="0.5")
# ax.minorticks_off()
ax_inset.minorticks_off()

plt.show()

plt.tight_layout()
plt.savefig("plots/132_v2.pdf", format="pdf")
plt.show()

# %% 1.4.2
result142, result142_std_dev, result143, result143_std_dev = generate_k1_data(N_arr, m=2, method="PA", iterations=1000,
                                                                              a=1.1)

# %% 1.4.2 & 1.4.3 save
with open("data/143_1000repeats_m=2.pkl", "wb") as f:
    pickle.dump((result143, result143_std_dev, result142, result142_std_dev), f)

# %% 1.4.2 & 1.4.3 load
with open("data/143_1000repeats_m=2.pkl", "rb") as f:
    result143, result143_std_dev, result142, result142_std_dev = pickle.load(f)

# %% 1.4.2 plot k_1
plt.errorbar(N_arr, result142, fmt=".", yerr=result142_std_dev / np.sqrt(1000), label="Empirical")
N_dense = np.arange(1, max(N_arr), 1)
plt.plot(N_dense, k1_pa_analytical(m=2, N=N_dense), "--", label="Theoretical")

# noinspection PyTupleAssignmentBalance
p, pcov = np.polyfit(np.log(N_arr), np.log(result142), 1, cov=True)
y_intercept = unc.ufloat(p[1], np.sqrt(pcov[1][1]))
gradient = unc.ufloat(p[0], np.sqrt(pcov[0][0]))
print(gradient)

x = np.linspace(1, max(N_arr), 100_000)
plt.plot(x, np.exp(p[1]) * x ** p[0], "--", label="Linear Regression")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("$k_1$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/142.pdf", format="pdf")
plt.show()

# %% 1.4.3 plot data collapse
for i in range(len(result143)):
    k = result143[i][0]
    k_1_empirical = result142[i]
    p_theory = p_pa_analytical(k, 2)
    plt.errorbar(k / k_1_empirical, result143[i][1] / p_theory,
                 yerr=result143_std_dev[i][1] / (p_theory * np.sqrt(1000)),
                 fmt=".", label=f"N = {N_arr[i]}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k/k^{em}_1$")
plt.ylabel(r"$p(k)/p^{th}\left( k \right)$")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/143.pdf", format="pdf")
plt.show()

# %%
plot_data_collapse("PA", result143, result143_std_dev, result142, m=2, repeats=1000)
plt.savefig("plots/143.pdf", format="pdf")

# %% 2.2
result22_mean = []
result22_std_dev = []
N = 1_000_000

for m in m_arr:
    temp = []
    for _ in range(50):
        ba = BarabasiAlbert(m + 1)
        ba.drive(N, m, "RA")
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=True)))

    temp = pad_logbin(temp)

    result22_mean.append(np.mean(temp, axis=0))
    result22_std_dev.append(np.std(temp, axis=0))

# %% 2.2 save
with open("data/result22.pkl", "wb") as f:
    pickle.dump((result22_mean, result22_std_dev), f)

# %% 2.2 load
with open("data/result22.pkl", "rb") as f:
    result22_mean, result22_std_dev = pickle.load(f)

# %% 2.2 plot
for i, (mean, std_dev) in enumerate(zip(result22_mean, result22_std_dev)):
    # propagate the standard deviations through the log using first order Taylor expansion
    plt.errorbar(mean[0], mean[1], yerr=std_dev[1] / np.sqrt(50), fmt=".", label=f"m = {m_arr[i]}", color=p[i], ms=3)
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

# %% 2.4.2
result242, result242_std_dev, result243, result243_std_dev = generate_k1_data(N_arr, m=2, method="RA", iterations=1000)

# %% 2.4.2 & 2.4.3 save
with open("data/243_1000repeats_m=2.pkl", "wb") as f:
    pickle.dump((result243, result243_std_dev, result242, result242_std_dev), f)

# %% 2.4.2 & 2.4.3 load
with open("data/243_1000repeats_m=2.pkl", "rb") as f:
    result243, result243_std_dev, result242, result242_std_dev = pickle.load(f)

# %% 2.4.2 plot
plt.errorbar(N_arr, result242, fmt=".", yerr=result242_std_dev/np.sqrt(1000), label="Empirical")
N_dense = np.arange(1, max(N_arr), 1)
plt.plot(N_dense, k1_ra_analytical(m=2, N=N_dense), "--", label="Theoretical")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("$k_1$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/242.pdf", format="pdf")
plt.show()

# %%
from logbin import logbin
N = 1_000_000
D = []
D_power_law = []

for i, m in enumerate(m_arr):
    ba = BarabasiAlbert(m + 1)
    ba.drive(N, m, method="RA")
    x, y = logbin(ba.node_degrees(), scale=1)
    ECDF = np.cumsum(y)
    CDF_theoretical = np.cumsum(p_ra_analytical(x, m=m))
    # CDF_theoretical = np.concatenate((np.zeros(m), CDF_theoretical))
    # CDF_theoretical = CDF_pa_analytical(np.arange(0, max(ba.node_degrees()) + 1), m=m)
    D.append(max(abs(ECDF - CDF_theoretical)))

D = np.array(D)
D_crit = D_threshold(0.05, m=N, n=N)

print(D-D_crit)
# %%

from logbin import logbin
N = 1_000_000
D = []

for i, m in enumerate(m_arr):
    ba = BarabasiAlbert(m + 1)
    ba.drive(N, m, method="RA")
    CDF_theoretical = np.cumsum(p_ra_analytical(np.arange(m, max(ba.node_degrees()+1)), m=m))
    CDF_theoretical = np.concatenate((np.zeros(m), CDF_theoretical))
    # CDF_theoretical = CDF_pa_analytical(np.arange(0, max(ba.node_degrees()) + 1), m=m)
    D.append(max(abs(ba.ECDF() - CDF_theoretical)))

D = np.array(D)
D_crit = D_threshold(0.05, m=N, n=N)

print(D_crit - D)

# %%
# plot_k1("RA", result242, result242_std_dev, m=2, repeats=1)

# %% 2.4.3 plot
for i in range(len(result243)):
    k = result243[i][0]
    k_1_empirical = result242[i]
    k_1_theory = k1_ra_analytical(m=2, N=N_arr[i])
    p_theory = p_ra_analytical(k, 2)
    plt.errorbar(k / k_1_theory, result243[i][1] / p_theory, yerr=result243_std_dev[i][1] / (p_theory * np.sqrt(100)),
                 fmt=".", label=f"N = {N_arr[i]}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$k/k^{em}_1$")
plt.ylabel(r"$p(k)/p^{th}\left( k \right)$")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/243.pdf", format="pdf")
plt.show()

# %%
plot_data_collapse("RA", result243, result243_std_dev, result242, m=2, repeats=1000)
plt.savefig("plots/243.pdf", format="pdf")

# %% 3.2
result32_mean = []
result32_std_dev = []
result32_SE = []
N = 100_0000
m32_arr = np.array([3, 9, 27, 81, 243])

for m in m32_arr:
    temp = []
    for _ in range(50):
        ba = BarabasiAlbert(m + 1)
        ba.drive(N, m, "MA")
        temp.append(list(ba.log_binned_degree_distribution(a=1.2, zeros=True, rm=True)))

    temp = pad_logbin(temp)

    result32_mean.append(np.mean(temp, axis=0))
    result32_std_dev.append(np.std(temp, axis=0))

# %%
N = 1_000_000
m32_arr = np.array([3, 9, 27, 81, 243])

powerlaw_MA_arr = np.zeros(len(m32_arr))
powerlaw_MA_std_dev = np.zeros(len(m32_arr))
powerlaw_MA_xmin = np.zeros(len(m32_arr))

for i, m in enumerate(m32_arr):
    ba = BarabasiAlbert(m + 1)
    ba.drive(N, m, "MA")
    # noinspection PyTypeChecker
    result = powerlaw.Fit(ba.node_degrees())
    powerlaw_MA_arr[i] = result.alpha
    powerlaw_MA_std_dev[i] = result.sigma
    powerlaw_MA_xmin[i] = result.xmin

# %% save 3.3.2 powerlaw
with open("data/powerlaw_MA_1mil.pkl", "wb") as f:
    pickle.dump((powerlaw_MA_arr, powerlaw_MA_std_dev), f)

# %% 3.3.2 load powerlaw
with open("data/powerlaw_MA_1mil.pkl", "rb") as f:
    powerlaw_MA_arr, powerlaw_MA_std_dev = pickle.load(f)

# %%
print(powerlaw_MA_std_dev)
# %% save
with open("data/result32.pkl", "wb") as f:
    pickle.dump((result32_mean, result32_std_dev), f)

# %% load
with open("data/result32.pkl", "rb") as f:
    result32_mean, result32_std_dev = pickle.load(f)

# %%
plt.rcParams.update({"legend.fontsize": 12})
m32_arr = np.array([3, 9, 27, 81, 243])

for i, (mean, std_dev) in enumerate(zip(result32_mean, result32_std_dev)):
    plt.errorbar(mean[0], mean[1], yerr=std_dev[1] / np.sqrt(50), fmt=".", label=f"m = {m32_arr[i]}", color=p[i], ms=3)
    k_dense = np.linspace(1, max(mean[0]), 1_000_000)
    plt.plot(k_dense, p_ma_analytical(k_dense, m=m32_arr[i]), "--", color=p[i], linewidth=0.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("$p(k)$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/332.pdf", format="pdf")
plt.show()

# %%
N_arr = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
result342, result342_std_dev, result343, result343_std_dev = generate_k1_data(N_arr, m=3, method="MA", iterations=1000,
                                                                              a=1.2)

# %%
with open("data/343_1000repeats_m=3.pkl", "wb") as f:
    pickle.dump((result343, result343_std_dev, result342, result342_std_dev), f)

# %%
with open("data/343_1000repeats_m=3.pkl", "rb") as f:
    result343, result343_std_dev, result342, result342_std_dev = pickle.load(f)

# %%

from logbin import logbin
D = []
m32_arr = np.array([3, 9, 27, 81, 243])

for i, m in enumerate(m32_arr):
    ba = BarabasiAlbert(m + 1)
    ba.drive(N, m, method="MA")
    x, y = logbin(ba.node_degrees(), scale=1)
    ECDF = np.cumsum(y)
    CDF_theoretical = np.cumsum(p_ma_analytical(x, m=m))
    # CDF_theoretical = np.concatenate((np.zeros(m), CDF_theoretical))
    # CDF_theoretical = CDF_pa_analytical(np.arange(0, max(ba.node_degrees()) + 1), m=m)
    D.append(max(abs(ECDF - CDF_theoretical)))

D = np.array(D)
D_crit = D_threshold(0.05, m=N, n=N)

print(D_crit - D)

# %%
plt.errorbar(N_arr, result342, fmt=".", yerr=result342_std_dev / np.sqrt(100), label="Empirical")
N_dense = np.arange(1, max(N_arr), 1)
# plt.plot(N_dense, k1_ra_analytical(m=2, N=N_dense), "--", label="Analytical")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("$k_1$")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/342.pdf", format="pdf")
plt.show()

# %% collapse

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6942, 4.016538), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
plt.subplots_adjust(hspace=0.05)

for i in range(len(result343)):
    k = result343[i][0]
    k_1_empirical = result342[i]
    # k_1_theory = k1_pa_analytical(m=2, N=N_arr[i])
    p_theory = p_ma_analytical(k, 3)
    # ax0.errorbar(k/k_1_empirical, result343[i][1]/p_theory, yerr=result343_std_dev[i][1]/(p_theory * np.sqrt(100)),
    #              fmt=".", label=f"N = {N_arr[i]}")
    ax0.errorbar(k / k_1_empirical, result343[i][1] / p_theory,
                 fmt=".", label=f"N = {N_arr[i]}")

    ax1.errorbar(k / k_1_empirical, np.zeros(len(k)),
                 yerr=result343_std_dev[i][1] / (p_theory * np.sqrt(100)),
                 fmt=".", label=f"N = {N_arr[i]}", ms=0, capsize=2)

ax0.set_xscale("log")
ax0.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlabel(r"$k/k^{em}_1$")
ax0.set_ylabel(r"$p(k)/p^{th}\left( k \right)$")
ax0.legend()
plt.tight_layout()
# plt.savefig("plots/343.pdf", format="pdf")
plt.show()

# %%
plot_data_collapse("MA", result343, result343_std_dev, result342, m=3, repeats=1000)
plt.rcParams.update({"legend.fontsize": 10})
plt.savefig("plots/343.pdf", format="pdf")
plt.show()
