from functools import partial
import os
from collections import ChainMap
from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh import Scene

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
    count_agree_pos = (m1_pos & m2_pos).sum()  # intersection
    count_pos = (m1_pos & m2_neg).sum() + (m1_neg & m2_pos).sum() + count_agree_pos
    if count_pos == 0:
        return 0
    return count_agree_pos / count_pos


def v_generator(perc=0.5):
    res = RESOLUTION[0] * RESOLUTION[1]
    return np.random.choice(res, int(res * perc), False)


if __name__ == "__main__":
    sil = raytrace_silhouette(get_transformed_scene(mesh), perc=1)
    plt.imshow(sil)
    plt.show()
