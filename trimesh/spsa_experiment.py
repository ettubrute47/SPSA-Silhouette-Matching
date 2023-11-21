from functools import partial, reduce
from typing import TypedDict, Optional
from collections import ChainMap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import trimesh

from problem import (
    mesh,
    get_transformed_scene,
    raytrace_silhouette,
    stochastic_jacaard_index,
)
from spsa import OptimSPSA, OptimFDSA, balanced_bernouli, segmented_uniform_sample


true_rotation = np.array([1, 0, 1])  # np.random.rand(3) * 2 * np.pi
true_translation = np.array([1, 1, 5])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*true_rotation), true_translation
)
true_sil = raytrace_silhouette(scene, perc=1)

# for initial perturbation, do spsa...

rotation = true_rotation + np.array([0.6, 0.1, 0.2])
translation = true_translation + np.array([-0.5, -0.5, -4])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*rotation), translation
)
sil = raytrace_silhouette(scene, perc=1)


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


def v_generator(perc=0.5):
    return np.random.choice(640 * 480, int(640 * 480 * perc), False)


theta_0 = np.append(rotation, translation)

scene = get_scene_from_theta(theta_0)
sil = raytrace_silhouette(scene, perc=0.5)
plt.imshow(sil + true_sil)
plt.show()

goal_theta = np.append(true_rotation, true_translation)

consts = dict(
    max_delta_theta=0.1, max_iter=30, alpha=1.0, num_approx=1, momentum=0.9, c0=1000
)
optim = OptimFDSA(theta_0, partial(get_loss_from_theta, perc=1), **consts)
losses, dists = optim.experiment(3, goal_theta=goal_theta)

fig, axs = plt.subplots(1, 2)
axs[0].plot(losses)
axs[1].plot(dists)

plt.show()
