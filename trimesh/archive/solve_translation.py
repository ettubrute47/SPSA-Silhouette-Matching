# %%
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
    v_generator,
)
from spsa import OptimSPSA, OptimFDSA, balanced_bernouli, segmented_uniform_sample

# %%

true_rotation = np.array([1, 0, 1])  # np.random.rand(3) * 2 * np.pi
true_translation = np.array([1, 1, 5])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*true_rotation), true_translation
)
true_sil = raytrace_silhouette(scene, perc=1)

# for initial perturbation, do spsa...

rotation = true_rotation + np.array([0.6, 0.1, 0.2])
translation = true_translation + np.array(
    [-0.5, -0.5, -0.5]
)  # np.array([-0.5, -0.5, -4])

scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*rotation), translation
)
sil = raytrace_silhouette(scene, perc=1)

plt.imshow(sil + true_sil)


# %%
def bounding_box(mask):
    # Get the x and y coordinates of non-zero values in the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([rmin, rmax, cmin, cmax])


def combine_bounding_boxes(bb1, bb2):
    return np.array(
        [
            min(bb1[0], bb2[0]),
            max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]),
            max(bb1[3], bb2[3]),
        ]
    )


def bbox_to_mask(bbox, shape):
    rmin, rmax, cmin, cmax = bbox
    arr = np.zeros(shape)
    arr[rmin:rmax, cmin:cmax] = 1
    return arr


def get_scale(bb):
    # returns the length of the diagonal of the box
    return np.sqrt((bb[1] - bb[0]) ** 2 + (bb[3] - bb[2]) ** 2)


def get_xy_shift(src_bb, tgt_bb):
    x = (tgt_bb[1] + tgt_bb[0]) - (src_bb[0] + src_bb[1])
    y = (tgt_bb[3] + tgt_bb[2]) - (src_bb[3] + src_bb[2])
    return np.array([x, y]) / 2


def shift_bb_xy(bb, dxy):
    return np.append(bb[:2] + dxy[0], bb[2:] + dxy[1]).astype(int)


bb = bounding_box(sil == 2)

bb_true = bounding_box(true_sil == 2)
bb_true_msk = bbox_to_mask(bounding_box(true_sil == 2), true_sil.shape)
plt.imshow(bbox_to_mask(bb, true_sil.shape) + bb_true_msk)

# %%

dxy = get_xy_shift(bb, bb_true)
plt.imshow(bbox_to_mask(shift_bb_xy(bb, dxy), true_sil.shape) + bb_true_msk)
# %%

# I want to first shift the z to the appropriate place, and the x, y given my estimated z.
dx = dxy[0] * true_translation[-1] / scene.camera.K[0, 0]
dy = dxy[1] * true_translation[-1] / scene.camera.K[1, 1]

new_est = translation - np.array([dx, dy, 0])
scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*rotation), new_est
)
sil = raytrace_silhouette(scene, perc=1)
bb = bounding_box(sil == 2)

bb_true = bounding_box(true_sil == 2)
bb_true_msk = bbox_to_mask(bounding_box(true_sil == 2), true_sil.shape)
plt.imshow(bbox_to_mask(bb, true_sil.shape) + bb_true_msk)
# %%

dxy = get_xy_shift(bb, bb_true)
plt.imshow(bbox_to_mask(shift_bb_xy(bb, dxy), true_sil.shape) + bb_true_msk)
# %%


def correct_translation(rotation, translation, num_steps=1, z_gain=1):
    corrected = np.array(translation)
    for i in range(num_steps):
        scene = get_transformed_scene(
            mesh, trimesh.transformations.euler_matrix(*rotation), corrected
        )
        sil = raytrace_silhouette(scene, perc=1)
        bb = bounding_box(sil == 2)

        bb_true = bounding_box(true_sil == 2)
        dxy = get_xy_shift(bb, bb_true)
        c = 0.1
        z_diff = corrected[-1] - corrected[-1] * (get_scale(bb) / get_scale(bb_true))
        Z = corrected[-1] - z_diff * z_gain
        Z = corrected[-1] * (
            z_gain * get_scale(bb) / get_scale(bb_true) + max(0, (1 - z_gain))
        )

        # I want to first shift the z to the appropriate place, and the x, y given my estimated z.
        dx = dxy[0] * Z / scene.camera.K[0, 0]
        dy = dxy[1] * Z / scene.camera.K[1, 1]

        corrected = corrected - np.array([dx, dy, 0]) * 1.1
        corrected[-1] = Z
    return corrected


# so in reality, when I inverse perspective I'll get f(z) = ... x and y is function of distance to me
corrected = correct_translation(rotation, translation, 1)
scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*rotation), corrected
)
sil = raytrace_silhouette(scene, perc=1)
bb = bounding_box(sil == 2)

bb_true = bounding_box(true_sil == 2)
bb_true_msk = bbox_to_mask(bounding_box(true_sil == 2), true_sil.shape)
plt.imshow(bbox_to_mask(bb, true_sil.shape) + bb_true_msk)

# %%

# now theta is just rotation, but we also are approximating translation implicitly
# so interestingly, there is an implicit part of theta... which we need to handle later...
implicit_translation = np.array(translation)
translation_history = [implicit_translation]


def update_implicit_translation(optim: OptimSPSA):
    global implicit_translation, translation_history
    rotation = optim.theta[:3]
    # this can happen in the same step that I do my rotation call...
    z_gain = 1 / (1 + optim.k) ** 0.2
    implicit_translation = correct_translation(
        rotation, implicit_translation, 1, optim.ak(optim.k - optim._params["A"]) * 1.5
    )
    translation_history.append(implicit_translation)


def get_scene_from_theta(theta: np.ndarray):
    # theta is rotation matrix, and translation matrix
    rotation = theta[:3]
    translation = implicit_translation
    rotation_matrix = trimesh.transformations.euler_matrix(*rotation)
    return get_transformed_scene(mesh, rotation_matrix, translation)


def get_loss_from_sil(sil):
    js = stochastic_jacaard_index(true_sil, sil)
    js = np.clip(js, 0.00001, 0.99999)
    # return -np.log(js)
    log_odds = np.log(1 - js) - np.log(js)
    return log_odds


def get_loss_from_theta(theta: np.ndarray, perc=0.5, v=None):
    scene = get_scene_from_theta(theta)
    sil = raytrace_silhouette(scene, perc=perc, inds=v)
    return get_loss_from_sil(sil)


consts = dict(
    max_delta_theta=0.05,
    max_iter=50,
    # alpha=1.0,
    num_approx=2,
    momentum=0.2,
    c0=1e-2,
)
optim = OptimFDSA(
    np.array(rotation),
    partial(get_loss_from_theta, perc=0.99),
    on_theta_update=update_implicit_translation,
    # v_fnc=v_generator,
    **consts
)

theta = optim.run()

"""

The final specialized function is as follows:

implicit is something that it doesn't update, but it knows about... it goes towards loss
At each step, approx gradient (given implicit)

decrease noise


I can also speed things up by increasing resolution over time... 

Missing still: 1 direction step

"""
plt.plot(optim._loss_history)
plt.show()
plt.plot(optim.rms_dist_from_truth(true_rotation))
plt.show()
plt.plot(np.array(translation_history) - true_translation)
plt.show()
plt.plot(optim._grad_history)
plt.show()

scene = get_scene_from_theta(optim.theta)
sil = raytrace_silhouette(scene, 1)
plt.imshow(sil + true_sil)
# %%
