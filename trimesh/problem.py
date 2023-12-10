from functools import partial
import os
from collections import ChainMap
from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import trimesh
from trimesh import Scene

from spsa import Box, OptimSA
from utils import bounding_box, get_scale, get_xy_shift, shift_mask, bbox_to_mask

model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../models/bunny.ply"
)
mesh = trimesh.load(model_path)

# Normalize the mesh using trimesh functionality
mesh.apply_translation(-mesh.centroid)
scale_factor = 1.0 / mesh.extents.max()
mesh.apply_scale(scale_factor)

RESOLUTION = [320, 320]
RESOLUTION = [128, 128]


def get_transformed_scene(
    mesh, rotation_transform=None, translation_vector=None
) -> Scene:
    transformed = mesh.copy()
    if rotation_transform is not None:
        transformed.apply_transform(rotation_transform)
    # transformed.apply_translation(translation_vector)
    # tt = transformed.apply_obb()  # i guess it works..
    scene = Scene()
    scene.add_geometry(
        transformed, geom_name="mesh"
    )  # , transform=transformation_matrix)
    if translation_vector is not None:
        scene.apply_translation(translation_vector)
    # scene.camera.resolution = [640, 480]
    scene.camera.resolution = RESOLUTION
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = [
        60,
        60,
    ]  # * (scene.camera.resolution / scene.camera.resolution.max())
    return scene


scene = get_transformed_scene(
    mesh, trimesh.transformations.euler_matrix(*[0, 0, 0]), [0, 0, 0]
)
origin, vectors, pixels = scene.camera_rays()

box = Box([0, 0, 0, -5, -5, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi, 5, 5, 10])


def raytrace_silhouette(scene, perc=0.5, inds=None):
    if perc == 1:
        origin = np.broadcast_to(
            trimesh.transformations.translation_from_matrix(scene.camera_transform),
            vectors.shape,
        )
        hits = scene.geometry["mesh"].ray.intersects_any(
            ray_origins=origin, ray_directions=vectors
        )
        return hits.reshape(scene.camera.resolution)[:, ::-1].astype(np.uint8) + 1

    w, h = scene.camera.resolution
    if inds is None:
        inds = np.random.choice(w * h, int(w * h * perc), False)

    origin = np.broadcast_to(
        trimesh.transformations.translation_from_matrix(scene.camera_transform),
        (inds.shape[0], 3),
    )
    # run the mesh- ray test
    hits = scene.geometry["mesh"].ray.intersects_any(
        ray_origins=origin, ray_directions=vectors[inds]
    )

    # find pixel locations of actual hits
    sub_pixels = pixels[inds]
    pixel_ray = sub_pixels[hits]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    a[sub_pixels[:, 0], sub_pixels[:, 1]] = 1
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = 2
    return a


## jacaards index for comparing two incomplete masks:
def stochastic_jacaard_index(m1: np.ndarray, m2: np.ndarray):
    """Where 2 means positive, 1 means negative, and 0 means not measured/unknown, I / U should only consider knowns"""
    m1_pos = m1 == 2
    m2_pos = m2 == 2
    m1_neg = m1 == 1
    m2_neg = m2 == 1
    # %pixels inside sil1 I know arent in sil2, and % pixels in sil2 i know arent in sil1
    count_agree_pos = (m1_pos & m2_pos).sum()  # intersection
    # count_pos = (m1_pos | m2_pos).sum()
    count_pos = (m1_pos & m2_neg).sum() + (m1_neg & m2_pos).sum() + count_agree_pos
    if count_pos == 0:
        return 0
    return count_agree_pos / count_pos


def v_generator(perc=0.5):
    res = RESOLUTION[0] * RESOLUTION[1]
    return np.random.choice(res, int(res * perc), False)


def get_scene_from_theta(theta: np.ndarray):
    # theta is rotation matrix, and translation matrix
    rotation = theta[:3]
    translation = theta[3:]
    rotation_matrix = trimesh.transformations.euler_matrix(*rotation)
    return get_transformed_scene(mesh, rotation_matrix, translation)


def get_sil_from_theta(theta, perc=0.5, v=None):
    scene = get_scene_from_theta(theta)
    return raytrace_silhouette(scene, perc=perc, inds=v)


class LossFnc:
    def __init__(self, goal_theta: np.ndarray, perc=0.5, viz=False, correct_xy=False):
        self.goal_theta = goal_theta
        self.perc = perc
        self.viz = viz
        self.correct_xy = correct_xy
        scene = get_scene_from_theta(goal_theta)
        self.true_sil = raytrace_silhouette(scene, perc=1.0)
        self.true_bb = bounding_box(self.true_sil == 2)

    def loss_from_sil(self, sil):
        js = stochastic_jacaard_index(self.true_sil, sil)
        js = np.clip(js, 0.00001, 0.99999)
        return -np.log(js)

    def __call__(self, theta, v=None):
        scene = get_scene_from_theta(theta)
        sil = raytrace_silhouette(scene, perc=self.perc, inds=v)
        if self.correct_xy:
            bb = bounding_box(sil == 2)
            shift = get_xy_shift(self.true_bb, bb)
            sil2 = shift_mask(sil, shift)
        else:
            sil2 = sil
        if self.viz:
            viz_theta(self.goal_theta)
            viz_sil(sil == 2, color="green")
            viz_sil(sil2 == 2, color="blue")
            plt.show()
        loss = self.loss_from_sil(sil2)
        if self.viz:
            print(loss)
        return loss

    def get_implict_update(self):
        # return TranslationEstimator(
        #     self.true_sil, true_z=self.goal_theta[-1]
        # ).on_theta_update
        return TranslationEstimator(self.true_sil).on_theta_update


# Generate a random configuration and then perturb it...


class TranslationEstimator:
    def __init__(self, true_sil: np.ndarray, true_z=None):
        self.bb_true = bounding_box(true_sil == 2)
        self.true_scale = get_scale(self.bb_true)
        self.true_z = true_z

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
            if self.true_z is not None:
                Z = self.true_z

            # I want to first shift the z to the appropriate place, and the x, y given my estimated z.
            dx = dxy[0] * Z / scene.camera.K[0, 0]
            dy = dxy[1] * Z / scene.camera.K[1, 1]

            corrected = corrected - np.array([dx, dy, 0]) * 1.1
            corrected[-1] = Z

        scene = get_transformed_scene(
            mesh, trimesh.transformations.euler_matrix(*rotation), corrected
        )
        sil = raytrace_silhouette(scene, perc=1)
        bb = bounding_box(sil == 2)

        dxy2 = get_xy_shift(bb, self.bb_true)
        # assert np.all(np.abs(dxy2) < np.abs(dxy))
        return corrected

    def on_theta_update(self, optim: OptimSA):
        theta = optim.box.unnormalize(optim.theta)
        rotation = theta[:3]
        translation = theta[3:]
        z_gain = optim.a_gain(optim.k - optim._params["A"]) / optim.a_gain(
            0 - optim._params["A"]
        )
        # z_gain = 0.1 / (optim.k + 1) ** 1.2
        corrected = self.correct_translation(rotation, translation, z_gain=z_gain)
        theta[3:] = corrected
        optim.thetas[-1] = optim.box.normalize(theta)


def viz_sil(sil, color="red", alpha=0.5):
    cmap = mcolors.ListedColormap(["none", color])
    plt.imshow(sil, alpha=alpha, cmap=cmap)


def viz_theta(theta, normalized=False, perc=1.0, color="red", alpha=0.5):
    if normalized:
        theta = box.unnormalize(theta)
    scene = get_scene_from_theta(theta)
    sil = raytrace_silhouette(scene, perc=perc)
    viz_sil(sil, color, alpha)


def random_xy_theta(theta):
    theta = np.array(theta)
    px, py = np.random.rand(2) - 0.5
    px *= RESOLUTION[0]
    py *= RESOLUTION[0]
    Z = theta[-1]
    X = -Z * px / scene.camera.K[0, 0]
    Y = -Z * py / scene.camera.K[0, 0]
    theta[-3] = X
    theta[-2] = Y
    return theta


def perturb_xy_theta(theta, mag=0.5):
    theta = np.array(theta)
    px, py = np.random.rand(2) * mag * RESOLUTION[0] - RESOLUTION[0] * mag / 2
    Z = theta[-1]
    X = -Z * px / scene.camera.K[0, 0]
    Y = -Z * py / scene.camera.K[0, 0]
    theta[-3] += X
    theta[-2] += Y
    return theta


if __name__ == "__main__":
    sil = raytrace_silhouette(get_transformed_scene(mesh), perc=1)
    # plt.imshow(sil)
    # plt.show()
    thetas = []
    perturbs = []
    np.random.seed(1337)
    for i in range(7):
        theta = box.sample()
        # I can randomize X and Y from image to make sure its still in there...
        theta = random_xy_theta(theta)
        mag = 0.15
        perturb_xy = perturb_xy_theta(np.array(theta), mag=mag)
        perturb = box.perturb(theta, mag=mag)
        perturb[-3:-1] = perturb_xy[-3:-1]
        # perturb = perturb_xy_theta(perturb, 0.2)
        sil1 = get_sil_from_theta(theta, perc=1)
        sil2 = get_sil_from_theta(perturb, perc=1)
        bb1 = bounding_box(sil1 == 2)
        bb2 = bounding_box(sil2 == 2)
        shift = get_xy_shift(bb1, bb2)
        sil3 = shift_mask(sil2, shift)
        bb3 = bounding_box(sil3 == 2)
        print(bb3, bb1, bb2)

        viz_theta(theta)
        viz_sil(sil2 == 2, color="green")
        # viz_sil(sil3 == 2, color="blue")
        # viz_sil(bbox_to_mask(bb3, sil.shape), color="blue")
        # viz_sil(bbox_to_mask(bb1, sil.shape), color="red")
        # plt.show()
        thetas.append(np.array(theta))
        perturbs.append(np.array(perturb))

    np.save("data/thetas2", np.array(thetas))
    np.save("data/perturbs2", np.array(perturbs))
    plt.show()
