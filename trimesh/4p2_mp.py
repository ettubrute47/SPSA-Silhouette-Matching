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
from spsa import OptimSPSA, OptimFDSA, Box, balanced_bernouli, segmented_uniform_sample
from problem import (
    mesh,
    get_transformed_scene,
    raytrace_silhouette,
    stochastic_jacaard_index,
)

true_rotation = np.array([1, 0, 1])  # np.random.rand(3) * 2 * np.pi
true_translation = np.array([1, 1, 5])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*true_rotation), true_translation
)
true_sil = raytrace_silhouette(scene, perc=1)

rotation = true_rotation + np.array([0.6, 0.1, 0.2])
translation = true_translation + np.array([-0.5, -0.5, -4])
# translation = true_translation + np.array([-0.5, -0.5, -0.5])

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
goal_theta = np.append(true_rotation, true_translation)

box = Box([0, 0, 0, -5, -5, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi, 5, 5, 10])


def run_optimization(*args):
    optim = OptimSPSA(
        np.array(theta_0),
        box,
        partial(get_loss_from_theta, perc=0.5),
        max_iter=200,
        max_delta_theta=0.05,
        loss_rng=10,
        c_std_scale=5,
    )
    optim.run()

    return (
        optim._loss_history,
        optim.rms_dist_from_truth(goal_theta),
        box.unnormalize(optim.thetas) - goal_theta,
    )


if __name__ == "__main__":
    import time
    import multiprocessing as mp
    import pickle

    num_trials = 12
    num_procs = 6
    # start = time.perf_counter()
    # for i in range(num_trials):
    #     run_optimization()
    # end = time.perf_counter()
    # avg = (end - start) / num_trials
    # print(f"Serial: {end - start:.2f} Total, Average {avg:.2f} sec per run\n")

    with mp.Pool(num_procs) as pool:
        print("Start MP")
        start = time.perf_counter()
        results = pool.map(run_optimization, range(num_trials))
    end = time.perf_counter()
    avg = (end - start) / num_trials
    print(f"MP: {end - start:.2f} Total, Average {num_procs * avg:.2f} sec per run")
    with open("data/4p2.pkl", "wb") as wtr:
        pickle.dump(results, wtr)
