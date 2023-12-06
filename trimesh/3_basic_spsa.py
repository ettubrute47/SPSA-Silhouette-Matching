# %%
from functools import partial, reduce
from typing import TypedDict, Optional
from collections import ChainMap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("darkgrid")
from spsa import OptimSPSA, Box, OptimFDSA, balanced_bernouli, segmented_uniform_sample

# %%
"""
Section 3 describes exploring SPSA in a basic fashion; 2d controlled movement

From initial playing around with the problem, there was an early issue in understanding the 
roles of the a and c gain, and the effect of noise on individual loss measurements and gradient estimates

When the noise increases, the gradient noise increases, but we can make up for that by looking across a larger gap
assuming that the gradient doesn't change too much

We can demonstrate the behavior with the 3 plots of moving theta

Figure 3.1


3.2 Automatic Configuration

For the parameters, we discuss how we implemented automatic configuration of the gains. Based on Spall
we have some simple methods; first is alpha = 0.602, gamma is ..., and A is 5-10%

This requires us specifying the maximum number of iterations we'd like to run

Then there is estimating a from the expected change in things, normalizing theta, and
choosing c. For this we just specify the max amount we'd like theta to move, and same with c

We discuss that in practice, choosing c to be 3* std seemed to be best, but in reality this parameter
likely can't be tuned simply like this

We show the performance for increasing levels of noise, as well as increasaing distances from the start
- it should show that it is relatively robust

I'd want to specify how small the steps I'd want to make by the end of it...

3.3 Additional Improvements

Talk about blocking, 2SPSA, momentum, other distributions, averaging, FDSA, Common Random Numbers

"""

# simple thing is 'root' finding, where loss is just the current value, and theta is x
goal_theta = np.array([1, 1])


def noise_source(noise_scale=1e-3):
    return np.random.normal(0, noise_scale, size=len(goal_theta))


def noisy_loss(theta, noise_scale=1e-3, v=None):
    if v is None:
        v = noise_source(noise_scale)
    diff = np.asarray(theta) - goal_theta + v
    return np.sqrt(diff @ diff)


theta_0 = np.array([0.5, 1.5])
# theta_0 = np.array(goal_theta)

box = Box(np.ones(2) * -3, np.ones(2) * 3)

# %%
optim = OptimSPSA(
    theta_0,
    box,
    partial(noisy_loss, noise_scale=1e-1),
    max_iter=200,
    loss_rng=6,
    max_delta_theta=0.01,
)
optim.run()
diffs = optim.rms_dist_from_truth(goal_theta)
diffs /= diffs[0]
plt.plot(diffs)


# %%
def plot_trail(optim: OptimSPSA, ax=None):
    if ax is None:
        ax = plt.gca()
    thetas = box.unnormalize(optim.thetas)
    cmap = plt.cm.coolwarm
    num_segments = len(thetas)
    for i in range(len(thetas) - 1):
        x1, y1 = thetas[i]
        x2, y2 = thetas[i + 1]
        ax.plot([x1, x2], [y1, y2], color=cmap(i / num_segments), marker="x", zorder=0)
    ax.scatter(*theta_0, color="k", marker="o", s=40, zorder=1)
    ax.scatter(*goal_theta, color="g", marker="o", s=40, zorder=1)


plot_trail(optim)
# %%
fig, axs = plt.subplots(1, 3, figsize=(10, 4))
optim = OptimSPSA(
    theta_0,
    box,
    partial(noisy_loss, noise_scale=1e-2),
    max_iter=1000,
    loss_rng=6,
    max_delta_theta=0.05,
    k0=800,
)
"""
Start with a = 0.1, show that c does have an effect, c=0.01 to c=0.1 reduces noise
But even with c=0.01, with a=0.01 we see it performs better...?
"""
print(optim._params)
optim.run(c0=1e-2, a0=0.1)
print(optim._params)
plot_trail(optim, axs[0])
optim.run(a0=0.1, c0=0.1)
plot_trail(optim, axs[1])
optim.run(c0=1e-2, a0=0.001)
print(optim._params)
plot_trail(optim, axs[2])

# %%
optim = OptimSPSA(
    theta_0,
    box,
    partial(noisy_loss, noise_scale=1e-2),
    max_iter=1000,
    loss_rng=6,
    # max_delta_theta=0.05,
)

optim.run()

plt.plot(optim.rms_dist_from_truth(goal_theta))


# %%


def setup_a_vs_c(optim: OptimSPSA, i):
    # normalize versus loss...
    # ang = np.random.rand() * 2 * np.pi
    # r = 1
    # theta_0 = np.array([np.cos(ang), np.sin(ang)]) * r
    # # optim.loss = partial(noisy_loss, noise_scale=noise[i])
    # optim.reset(theta_0)
    # optim.set_params(max_delta_theta=3 * (r + 3) ** 0.8 / 100)
    pass


def get_metrics(optim: OptimSPSA, i):
    return (optim._loss_history, optim.rms_dist_from_truth(goal_theta))


def to_df(arr, name="Value"):
    run_id = np.arange(arr.shape[0])
    df = pd.DataFrame(arr, columns=[i for i in range(arr.shape[1])])
    df["Run"] = run_id
    df_melted = df.melt(id_vars=["Run"], var_name="Rep", value_name=name)
    return df_melted


def experiment_3p1(num_reps=100):
    optim = OptimSPSA(
        np.array(goal_theta),
        box,
        partial(noisy_loss, noise_scale=1e-2),
        loss_rng=20,
        max_iter=1000,
        k0=900,
        # max_delta_theta=0.1,
    )
    metric_dfs = []
    params = [
        {"a0": 1e-1, "c0": 1e-2},
        {"a0": 1e-1, "c0": 1e-1},
        {"a0": 1e-2, "c0": 1e-2},
    ]
    for i, param in enumerate(params):
        optim.set_params(**param)
        losses, dists = optim.custom_experiment(
            num_reps, setup_a_vs_c, get_metrics=get_metrics
        )
        dist_df = to_df(dists[1:])
        loss_df = to_df(losses, "Loss")
        loss_df["Params"] = f"$a_0 = {param['a0']}$ $c_0 = {param['c0']}$"
        loss_df["Experiment"] = i
        loss_df["Distance"] = dist_df["Value"]
        metric_dfs.append(loss_df)

    return pd.concat(metric_dfs, ignore_index=True)


df = experiment_3p1(2)
# %%
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(df, x="Run", y="Distance", hue="Params", ax=ax, n_boot=500)
ax.set_ylabel(r"$|\hat{\theta}-\theta^*|$")
sns.move_legend(ax, "center right")
# %%

"""
Figure 3.2 

Random draw from start with forced magnitude distance, 

And increasing noise

max travel... how far does it actually get? 
"""


num_trials = 10
r0 = np.linspace(2, 4, num_trials)
noise = np.logspace(-3, 0, num_trials)


def setup(optim: OptimSPSA, i):
    # normalize versus loss...
    ang = np.random.rand() * 2 * np.pi
    r = 1
    theta_0 = np.array([np.cos(ang), np.sin(ang)]) * r
    # optim.loss = partial(noisy_loss, noise_scale=noise[i])
    optim.reset(theta_0)
    optim.set_params(max_delta_theta=3 * (r + 3) ** 0.8 / 100)


def get_metrics(optim: OptimSPSA, i):
    return (optim._loss_history, optim.rms_dist_from_truth(goal_theta))


# optim = OptimSPSA(
#     theta_0, partial(noisy_loss, noise_scale=1e-1), max_iter=200, max_delta_theta=0.5
# )
# losses, dists = optim.custom_experiment(num_trials, setup, get_metrics)


def to_df(arr, name="Value"):
    run_id = np.arange(arr.shape[0])
    df = pd.DataFrame(arr, columns=[i for i in range(arr.shape[1])])
    df["Run"] = run_id
    df_melted = df.melt(id_vars=["Run"], var_name="Rep", value_name=name)
    return df_melted


def outer_experiment(num_trials, num_reps):
    optim = OptimSPSA(
        theta_0,
        partial(noisy_loss, noise_scale=1e-1),
        max_iter=200,
        max_delta_theta=0.1,
        c_std_scale=5,
    )
    metric_dfs = []
    noise = np.array([1e-3, 1e-2, 1e-1, 1])
    noise = np.sort(np.append(noise, noise[1:] / 2))
    num_trials = len(noise)
    for i, n in enumerate(noise):
        optim.loss = partial(noisy_loss, noise_scale=n)
        losses, dists = optim.custom_experiment(num_reps, setup, get_metrics)
        dist_df = to_df(dists[1:])
        loss_df = to_df(losses, "Loss")
        loss_df["noise"] = np.log10(n)
        loss_df["Experiment"] = i
        loss_df["Distance"] = dist_df["Value"]
        metric_dfs.append(loss_df)

    return pd.concat(metric_dfs, ignore_index=True)


loss_df = outer_experiment(5, 100)

# loss_df = to_df(losses)
# loss_df["noise"] = loss_df["Experiment"].map(dict(enumerate(noise)))
# # loss_df["Normal Loss"] = loss_df["Value"] / loss_df["r0"]
# # sns.lineplot(loss_df, x="Run", y="Normal Loss", hue="r0")
# sns.lineplot(to_df(losses, "Loss"), x="Run", y="Loss")
# plt.show()
# sns.lineplot(to_df(dists, "Distance"), x="Run", y="Distance")
# plt.show()

# %%
# sns.lineplot(loss_df, x="Run", y="Loss", hue="noise")
# plt.show()
sns.lineplot(loss_df, x="Run", y="Distance", hue="noise", n_boot=200)
# %%
"""
We see that the algorithm convergences close to the same time for different noise levels, 
well within the 100 step threshold...

For large noise it converges at a high loss average... we should look at distance honestly...
"""

sns.lineplot(loss_df, x="Run", y="Loss", hue="noise", n_boot=200)
# %%
