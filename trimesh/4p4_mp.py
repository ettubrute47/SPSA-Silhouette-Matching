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
from problem import box, LossFnc, viz_theta

thetas = np.load("data/thetas2.npy")
theta_0s = np.load("data/perturbs2.npy")
num_trials_per = 2


def run_optimization(*args, viz=False):
    i = int(args[0] / num_trials_per)
    if i == 4:
        i = 3
    loss = LossFnc(thetas[i], viz=False, correct_xy=False, perc=0.5)
    theta_0 = theta_0s[i]
    optim = OptimSPSA(
        np.array(theta_0),
        box,
        loss,
        max_iter=200,
        loss_rng=10,
        c_std_scale=4,
        # max_delta_theta=0.008,
        on_theta_update=loss.get_implict_update(),
        implicit_theta_mask=[1, 1, 1, 0, 0, 0],
    )
    optim.run()

    return (
        i,
        optim._loss_history,
        optim.rms_dist_from_truth(thetas[i]),
        box.unnormalize(optim.thetas) - thetas[i],
    )


if __name__ == "__main__":
    import time
    import multiprocessing as mp
    import pickle

    # viz_theta(thetas[2])
    # viz_theta(theta_0s[2], color="green")
    # plt.show()

    num_trials = num_trials_per * len(thetas)
    num_procs = 6
    # start = time.perf_counter()
    # for i in range(num_trials):
    #     print(i)
    #     res = run_optimization(i, viz=i == 2)
    #     print(res[1][-1])
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
    with open("data/4p4p5.pkl", "wb") as wtr:
        pickle.dump(results, wtr)
