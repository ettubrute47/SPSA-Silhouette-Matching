# %%
import cProfile
from collections import ChainMap
import multiprocessing as mp

import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from numba import jit

from mypar import pzip

from projection import (
    generate_cylinder_points,
    project_3d_to_2d,
    draw_silhouette,
    draw_filled_silhouette,
    compute_iou,
    cv2_imshow,
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler_angles,
)

from optim import optim_spsa, c_gain, a_gain

# %%


@jit(nopython=True)
def jaccard_index(mask1, mask2):
    # Directly compute the intersection and union using loops
    intersection = 0
    union = 0
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] or mask2[i, j]:
                union += 1
            if mask1[i, j] and mask2[i, j]:
                intersection += 1

    assert union > 0
    return intersection / union


class Cylinder:
    def __init__(self, K, radius, height, num_points, orientation=None, position=None):
        self.K = K
        self.radius = radius
        self.height = height
        self.num_points = num_points
        self.orientation = (
            np.array([0, 0, 0], float)
            if orientation is None
            else np.asarray(orientation)
        )
        self.position = (
            np.array([0, 0, 15]) if position is None else np.asarray(position)
        )
        self.cylinder_3d = generate_cylinder_points(radius, height, num_points)
        self._R = euler_to_rotation_matrix(*self.orientation)
        self._Rkey = self.orientation.copy()

        # Camera and scene properties
        self.K = K

    def copy(self, **kwargs):
        return Cylinder(
            **ChainMap(
                kwargs,
                {
                    "K": self.K.copy(),
                    "radius": self.radius,
                    "height": self.height,
                    "num_points": self.num_points,
                    "orientation": self.orientation.copy(),
                    "position": self.position.copy(),
                },
            )
        )

    @property
    def R(self):
        if not np.array_equal(self._Rkey, self.orientation):
            self._R = euler_to_rotation_matrix(*self.orientation)
            self._Rkey = self.orientation.copy()
        return self._R

    @property
    def translation(self):
        p = self.position.copy()
        p[1] -= self.height / 2
        return np.array([p]).T

    @property
    def image_shape(self):
        return (int(self.K[0, 2] * 2), int(self.K[1, 2] * 2))

    def silhouette(self):
        cylinder_2d = project_3d_to_2d(
            self.cylinder_3d, self.R, self.translation, self.K
        )
        return draw_filled_silhouette(cylinder_2d, *self.image_shape)

    def wireframe(self):
        cylinder_2d = project_3d_to_2d(
            self.cylinder_3d, self.R, self.translation, self.K
        )
        return draw_silhouette(cylinder_2d, *self.image_shape)

    def plot(self, filled=False):
        mask = self.wireframe() if filled is False else self.silhouette()
        cv2_imshow(mask)


# %%
def create_cylinder_from_unity():
    with open("cameraData.json", "r") as f:
        data = json.load(f)

    # Extract data
    f = data["K"]["f"]
    aspect_ratio = data["K"]["aspectRatio"]
    px = data["K"]["principalPoint"]["x"]
    py = data["K"]["principalPoint"]["y"]
    rotation_quaternion = [
        data["R"]["x"],
        data["R"]["y"],
        data["R"]["z"],
        data["R"]["w"],
    ]
    t = np.array([data["t"]["x"], data["t"]["y"], data["t"]["z"]])
    cylinder_radius = data["cylinderDimensions"]["cylinderRadius"]
    cylinder_height = data["cylinderDimensions"]["cylinderHeight"]

    R = quaternion_to_rotation_matrix(rotation_quaternion)
    orientation = rotation_matrix_to_euler_angles(R)

    K = np.array([[f, 0, px], [0, f * aspect_ratio, py], [0, 0, 1]])

    cylinder = Cylinder(
        K, cylinder_radius, cylinder_height, 30, np.array(orientation), t
    )
    cylinder.position[1] *= -1
    cylinder.orientation[0] += np.pi / 2
    return cylinder


# %%
cylinder = create_cylinder_from_unity()

# %%
mask = cylinder.silhouette()
mask

# %%
random_cylinder = cylinder.copy(orientation=np.random.rand(3) * 2 - 1)

# %%
jaccard_index(cylinder.silhouette(), random_cylinder.silhouette())


# %%
def loss(test_cylinder: Cylinder):
    overlap = jaccard_index(cylinder.silhouette(), test_cylinder.silhouette())
    if overlap == 0:
        return 100
    return min(-np.log(overlap), 100)


loss(random_cylinder)

# %%
# theta is orientation AND position, orientation is 0, 2*np.pi, and position is -2, 2, and ...
theta_bounds = np.array([np.pi / 2, np.pi, np.pi, 2, 2, 10])  # from theta_0
theta_0 = np.array([*random_cylinder.orientation, *random_cylinder.position])
c0 = theta_bounds / 20

# %% [markdown]
# So fixing everything except orientation is jumbled
#
# I need to define the range of orientation values that are possible, let's just say
# -np.pi/2 to np.pi/2
# -np.pi to np.pi * 2
#
# Now I want to standardize theta to scale that range to 1
#
# And then I want to sample N points in those dim spaces, and run spsa for each until we get to a local minima


# %%
def get_cylinder(theta):
    bounds = np.array([np.pi, 2 * np.pi, 2 * np.pi])
    return cylinder.copy(orientation=np.asarray(theta) * bounds)


def loss_norm(theta):
    return loss(get_cylinder(theta))


loss_norm(np.zeros(3))

# %%
optim = optim_spsa(np.zeros(3), loss_norm, a_gain(0.001, 1000), c_gain(0.001))

# %%
if __name__ == "__main__":
    for i, theta in zip(range(100), optim):
        print(loss_norm(theta))


# %%
def bounding_box(mask):
    # Get the x and y coordinates of non-zero values in the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def combine_bounding_boxes(bb1, bb2):
    return (
        min(bb1[0], bb2[0]),
        max(bb1[1], bb2[1]),
        min(bb1[2], bb2[2]),
        max(bb1[3], bb2[3]),
    )


def get_bbox_area(bb):
    rmin, rmax, cmin, cmax = bb
    return (rmax - rmin) * (cmax - cmin)


def norm_bbox_area(mask):
    area = get_bbox_area(bounding_box(mask))
    return area / mask.size


combine_bounding_boxes(bounding_box(random_cylinder.silhouette()), bounding_box(mask))

norm_bbox_area(mask)


# 1.0 means taking up entire space AND 0 overlap
# iou = 1 when i == u, so area * (1 - iou)
# 0.0 means perfect overlap
# issue is if disappear... it penalizes being 'big'
# what if I get bbox and penalize amount of 0 inside it relative to size of the object?
# this would encourage it to get big lol, but thats ok, b/c when it gets big itll start to intersect
# and when it starts to intersect its score will go down...
# or maybe I do iterative tasking - when intersection is 0, use bounding box score, when its not, use other...
# ok I'll do that. Problem is in boundary condition... so when iou is 0, use other, but when iou approaches 1, just use that
# normalize score for both
# OR: just random select until I get a point where there is an intersection


# %% [markdown]
# So it looks like it worked for orientation of 0... try for other orientations?

# %%


def optim_starter():
    return optim_spsa(np.random.rand(3), loss_norm, a_gain(0.001, 200), c_gain(0.001))


# pr = cProfile.Profile()
# pr.enable()

if __name__ == "__main__":
    print("\n\n=========================== Starting paralell ==============")
    import time

    MP = True
    NUM_ITERS = 1000
    NUM_POINTS = 6
    optim_iterators = []
    for i in range(NUM_POINTS):
        optim_iterators.append(optim_starter)

    start_time = time.perf_counter()
    if MP:
        with pzip(*optim_iterators, buffer=10) as piter:
            for i, thetas in zip(range(NUM_ITERS), piter):
                losses = list(map(loss_norm, thetas))
                print(
                    f"Iter {i}: "
                    + " ".join([np.format_float_positional(ti, 3) for ti in losses])
                )
    else:
        piter = [fnc() for fnc in optim_iterators]
        for i, *thetas in zip(range(NUM_ITERS), *piter):
            losses = list(map(loss_norm, thetas))
            print(
                f"Iter {i}: "
                + " ".join([np.format_float_positional(ti, 3) for ti in losses])
            )

    print("Duration: ", time.perf_counter() - start_time)

# pr.disable()
# pr.dump_stats("output/jit_profile_results.prof")
