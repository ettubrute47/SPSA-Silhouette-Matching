import numpy as np
import PIL
import trimesh

if __name__ == "__main__":
    # test on a sphere mesh
    mesh = trimesh.primitives.Sphere()

    # create some rays
    ray_origins = np.array([[0, 0, -5], [2, 2, -10]])
    ray_directions = np.array([[0, 0, 1], [0, 0, 1]])

    # scene will have automatically generated camera and lights
    scene = mesh.scene()

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [640, 480]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = [
        60,
        60,
    ]  # * (scene.camera.resolution / scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origin, vectors, pixels = scene.camera_rays()
    # if len(origin) > 10_000:
    #     print(len(origin))
    #     origin = ray_origins

    # intersects_location requires origins to be the same shape as vectors
    # origins = np.tile(np.expand_dims(origin, 0), (len(vectors), 1))

    # run the mesh- ray test
    points, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origin, ray_directions=vectors, multiple_hits=False
    )

    # you could also do this against the single camera Z vector
    depth = trimesh.util.diagonal_dot(points - origin[index_ray], vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    # create a PIL image from the depth queries
    img = PIL.Image.fromarray(a)

    img.show()

    # # stack rays into line segments for visualization as Path3D
    # ray_visualize = trimesh.load_path(
    #     np.hstack((origin, origin + vectors)).reshape(-1, 2, 3)
    # )

    # # make mesh transparent- ish
    # mesh.visual.face_colors = [100, 100, 100, 100]

    # # create a visualization scene with rays, hits, and mesh
    # scene = trimesh.Scene([mesh, ray_visualize, trimesh.points.PointCloud(locations)])

    # # display the scene
    # scene.show()
