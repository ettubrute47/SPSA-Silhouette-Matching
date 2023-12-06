# %%
from functools import partial, reduce
from typing import TypedDict, Optional
from collections import ChainMap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import trimesh

sns.set_style("darkgrid")
from spsa import OptimSPSA, Box, OptimFDSA, balanced_bernouli, segmented_uniform_sample
from problem import (
    mesh,
    get_transformed_scene,
    raytrace_silhouette,
    stochastic_jacaard_index,
    v_generator,
)
from utils import TranslationEstimator

# %%
"""
"""
true_rotation = np.array([1, 0, 1])  # np.random.rand(3) * 2 * np.pi
true_translation = np.array([1, 1, 5])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*true_rotation), true_translation
)
true_sil = raytrace_silhouette(scene, perc=1)

# rotation = true_rotation + np.array([0.3, 1.5, 0.2])
rotation = true_rotation + np.array([0.6, 0.1, 0.2])
translation = true_translation + np.array([-0.5, -0.5, -4])
# translation = true_translation + np.array([-0.5, -0.5, -0.5])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*rotation), translation
)
sil = raytrace_silhouette(scene, perc=1)
plt.imshow(sil + true_sil)

# %%


def get_scene_from_theta(theta: np.ndarray):
    # theta is rotation matrix, and translation matrix
    rotation = theta[:3]
    translation = theta[3:]
    rotation_matrix = trimesh.transformations.euler_matrix(*rotation)
    return get_transformed_scene(mesh, rotation_matrix, translation)


def get_loss_from_sil(sil):
    js = stochastic_jacaard_index(true_sil, sil)
    js = np.clip(js, 0.00001, 0.99999)
    return -np.log(js)
    log_odds = np.log(1 - js) - np.log(js)
    return log_odds


def get_loss_from_theta(theta: np.ndarray, perc=0.5, v=None):
    scene = get_scene_from_theta(theta)
    sil = raytrace_silhouette(scene, perc=perc, inds=v)
    return get_loss_from_sil(sil)


theta_0 = np.append(rotation, translation)
goal_theta = np.append(true_rotation, true_translation)
box = Box([0, 0, 0, -5, -5, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi, 5, 5, 10])


def setup_a_vs_c(optim: OptimSPSA, i):
    pass


def get_metrics(optim: OptimSPSA, i):
    return (optim._loss_history, optim.rms_dist_from_truth(goal_theta), optim.thetas)


def to_df(arr, name="Value"):
    run_id = np.arange(arr.shape[0])
    df = pd.DataFrame(arr, columns=[i for i in range(arr.shape[1])])
    df["Run"] = run_id
    df_melted = df.melt(id_vars=["Run"], var_name="Rep", value_name=name)
    return df_melted


implicit_translation = TranslationEstimator(true_sil)


def experiment_4p2(num_reps=100):
    optim = OptimSPSA(
        np.array(theta_0),
        box,
        partial(get_loss_from_theta, perc=0.5),
        max_iter=50,
        loss_rng=20,
        # max_delta_theta=0.1,
        c_std_scale=5,
        on_theta_update=implicit_translation.on_theta_update,
        implicit_theta_mask=[1, 1, 1, 0, 0, 0],
    )
    metric_dfs = []
    percs = np.array([0.05, 0.1, 0.25, 0.5, 0.8, 1.0])
    percs = np.array([0.3, 0.5, 0.8])
    for i, perc in enumerate(percs):
        print(f"Experiment {i}")
        optim.loss = partial(get_loss_from_theta, perc=perc)
        # optim.v_fnc = partial(v_generator, perc=perc)
        losses, dists, thetas = optim.custom_experiment(
            num_reps, setup_a_vs_c, get_metrics=get_metrics
        )
        dist0 = np.linalg.norm(box.normalize(theta_0) - box.normalize(goal_theta))
        print(optim._params)
        dist_df = to_df(dists[1:])
        loss_df = to_df(losses, "Loss")
        loss_df["Percent"] = perc
        loss_df["Experiment"] = i
        loss_df["Distance"] = dist_df["Value"] / dist0
        metric_dfs.append(loss_df)

    return pd.concat(metric_dfs, ignore_index=True)


df = experiment_4p2(20)
# %%
dist0 = np.linalg.norm(box.normalize(theta_0) - box.normalize(goal_theta))
df2 = df.copy()
df2["Distance"] /= dist0
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(df2, x="Run", y="Distance", hue="Percent", ax=ax, n_boot=500)
ax.set_ylabel(r"$|\hat{\theta}-\theta^*|$")
sns.move_legend(ax, "center right")
plt.show()
sns.histplot(df2[df2.Run == 49], x="Distance", hue="Percent")
# %%

optim = OptimSPSA(
    np.array(theta_0),
    box,
    partial(get_loss_from_theta, perc=0.5),
    max_iter=50,
    num_approx=1,
    loss_rng=20,
    # max_delta_theta_scale=0.1,
    c_std_scale=5,
    # theta_smooth=10,
    on_theta_update=implicit_translation.on_theta_update,
    implicit_theta_mask=[1, 1, 1, 0, 0, 0],
    # alpha=0.8,
)
theta = box.unnormalize(optim.run())

scene = get_scene_from_theta(theta)
sil = raytrace_silhouette(scene, 1.0)

plt.imshow(sil)
plt.show()
plt.plot(optim.thetas - box.normalize(goal_theta))
plt.show()

# %%
losses, dists, thetas = optim.custom_experiment(
    1, setup_a_vs_c, get_metrics=get_metrics
)

dfs = []
lbls = ["yaw", "pitch", "roll", "x", "y", "z"]
for i, theta_i in enumerate(thetas):
    dfs.append(to_df(thetas[i] - goal_theta[i]))
    dfs[-1]["Component"] = lbls[i]
df = pd.concat(dfs, ignore_index=True)
df

theta = optim.theta
scene = get_scene_from_theta(theta)
sil = raytrace_silhouette(scene, 1.0)

plt.imshow(sil + true_sil)
plt.show()
sns.lineplot(df, x="Run", y="Value", hue="Component")
plt.show()
# %%

# %%
