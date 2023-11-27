import numpy as np
import trimesh

from problem import (
    mesh,
    get_transformed_scene,
    raytrace_silhouette,
)


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


class TranslationEstimator:
    def __init__(self, true_sil: np.ndarray):
        self.bb_true = bounding_box(true_sil == 2)
        self.true_scale = get_scale(self.bb_true)

    def correct_translation(self, rotation, translation, num_steps=1, z_gain=1):
        corrected = np.array(translation)
        for i in range(num_steps):
            scene = get_transformed_scene(
                mesh, trimesh.transformations.euler_matrix(*rotation), corrected
            )
            sil = raytrace_silhouette(scene, perc=1)
            bb = bounding_box(sil == 2)

            dxy = get_xy_shift(bb, self.bb_true)
            c = 0.1
            z_diff = corrected[-1] - corrected[-1] * (
                get_scale(bb) / get_scale(self.bb_true)
            )
            Z = corrected[-1] - z_diff * z_gain
            Z = corrected[-1] * (
                z_gain * get_scale(bb) / self.true_scale + max(0, (1 - z_gain))
            )

            # I want to first shift the z to the appropriate place, and the x, y given my estimated z.
            dx = dxy[0] * Z / scene.camera.K[0, 0]
            dy = dxy[1] * Z / scene.camera.K[1, 1]

            corrected = corrected - np.array([dx, dy, 0]) * 1.1
            corrected[-1] = Z
        return corrected

    def on_theta_update(self, optim):
        rotation = optim.thetas[-1][:3]
        translation = optim.thetas[-1][3:]
        z_gain = optim.a_gain(optim.k - optim._params["A"]) * 1.5
        corrected = self.correct_translation(rotation, translation, z_gain=z_gain)
        optim.thetas[-1][3:] = corrected
