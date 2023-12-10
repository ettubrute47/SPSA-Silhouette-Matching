"""
Load meshes from any file format supported by trimesh package. Install trimesh and
required packages (e.g. networkx to read GLTF) with pip:

   pip install trimesh networkx

Materials are assigned by mesh names.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse, m_dispersive_glass

import trimesh


def main():
    rt = TkOptiX(devices=[0])  # create and configure, show the window later

    rt.set_param(max_accumulation_frames=500)  # accumulate up to 100 frames
    rt.set_uint("path_seg_range", 6, 12)  # allow some more ray segments
    rt.set_background(0)  # black background
    rt.set_ambient([0.1, 0.12, 0.15])  # some ambient light
    # rt.set_param(light_shading="Hard")        # nice, accurate caustics, but slower convergence

    exposure = 1
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Gamma")  # gamma correction

    # setup materials:
    m_diffuse["VarFloat3"] = {"base_color": [0.85, 0.87, 0.89]}
    rt.update_material("diffuse", m_diffuse)

    m_dispersive_glass["VarFloat3"]["base_color"] = [100, 110, 120]
    rt.setup_material("glass", m_dispersive_glass)

    # read the scene:
    rt.load
    mesh = trimesh.load("../models/soda_can.glb", force="mesh")
    rt.set_mesh("can", mesh.vertices, mesh.faces)

    # camera and light
    rt.setup_light("light1", pos=[6, 7.5, -15], color=30, radius=2)
    rt.setup_camera(
        "cam1", eye=[-2, 5, -10], target=[-0.75, 1.4, 5], fov=23, glock=True
    )

    rt.start()
    print("done")


if __name__ == "__main__":
    main()