# %%
from functools import partial, reduce
from typing import TypedDict, Optional
from collections import ChainMap
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from spsa import OptimSPSA, Box, OptimFDSA, balanced_bernouli, segmented_uniform_sample

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


"""
Create an optim and run it - and return results from it...

"""


def run_optimization(*args):
    optim = OptimSPSA(
        np.array(goal_theta),
        Box(np.ones(2) * -3, np.ones(2) * 3),
        partial(noisy_loss, noise_scale=1e-2),
        loss_rng=20,
        max_iter=1000,
        k0=900,
        # max_delta_theta=0.1,
    )
    optim.run()
    return (optim._loss_history, optim.rms_dist_from_truth(goal_theta))


if __name__ == "__main__":
    import time

    num_trials = 100
    start = time.perf_counter()
    for i in range(num_trials):
        run_optimization()
    end = time.perf_counter()
    avg = (end - start) / num_trials
    print(f"Serial: {end - start:.2f} Total, Average {avg:.2f} sec per run\n")

    with mp.Pool(6) as pool:
        print("Start MP")
        start = time.perf_counter()
        results = pool.map(run_optimization, range(num_trials))
    end = time.perf_counter()
    avg = (end - start) / num_trials
    print(f"MP: {end - start:.2f} Total, Average {avg:.2f} sec per run")

# def experiment_3p1(num_reps=100):
#     for i, param in enumerate(params):
#         optim.set_params(**param)
#         losses, dists = optim.custom_experiment(
#             num_reps, setup_a_vs_c, get_metrics=get_metrics
#         )
#         dist_df = to_df(dists[1:])
#         loss_df = to_df(losses, "Loss")
#         loss_df["Params"] = f"$a_0 = {param['a0']}$ $c_0 = {param['c0']}$"
#         loss_df["Experiment"] = i
#         loss_df["Distance"] = dist_df["Value"]
#         metric_dfs.append(loss_df)

#     return pd.concat(metric_dfs, ignore_index=True)


# df = experiment_3p1(2)
#     with mp.Pool(6) as pool:
#         for beta in betas:
#             print("Beta ", beta)
#             results = pool.map(partial(experiment_binary, beta, num_trials=100), percs)
#             result_array.append(results)
#     result_array = np.array(result_array)
