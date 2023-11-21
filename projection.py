import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_cylinder_points(radius, height, num_points):
    # Generate points for the base circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    base_x = radius * np.cos(theta)
    base_y = radius * np.sin(theta)
    base_z = np.zeros_like(base_x)

    # Generate points for the top circle
    top_x = base_x
    top_y = base_y
    top_z = np.ones_like(base_x) * height

    # Combine the points
    cylinder_points = np.vstack(
        (
            np.column_stack((base_x, base_y, base_z)),
            np.column_stack((top_x, top_y, top_z)),
        )
    )

    return cylinder_points


def project_3d_to_2d(points_3d, R, t, K):
    # Create a transformation matrix
    transformation_matrix = np.hstack((R, t))

    # Convert points to homogeneous coordinates
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project the points to 2D
    points_2d = np.dot(K, np.dot(transformation_matrix, points_3d_homo.T)).T
    points_2d = (points_2d / points_2d[:, 2, None])[
        :, :2
    ]  # Convert to inhomogeneous coordinates

    return points_2d


def draw_silhouette(points_2d, width, height):
    canvas = np.zeros((height, width), dtype=np.uint8)

    num_points = points_2d.shape[0] // 2
    for i in range(num_points):
        # Connect points in the base circle
        cv2.line(
            canvas,
            tuple(points_2d[i].astype(int)),
            tuple(points_2d[(i + 1) % num_points].astype(int)),
            255,
            1,
        )

        # Connect points in the top circle
        cv2.line(
            canvas,
            tuple(points_2d[i + num_points].astype(int)),
            tuple(points_2d[(i + 1) % num_points + num_points].astype(int)),
            255,
            1,
        )

        # Connect base and top points
        cv2.line(
            canvas,
            tuple(points_2d[i].astype(int)),
            tuple(points_2d[i + num_points].astype(int)),
            255,
            1,
        )

    return canvas


def draw_filled_silhouette(points_2d, width, height):
    canvas = np.zeros((height, width), dtype=np.uint8)

    num_points = points_2d.shape[0] // 2

    # Create lists to store the 2D points for the top and base circles
    base_circle = []
    top_circle = []

    for i in range(num_points):
        base_circle.append(points_2d[i].astype(int))
        top_circle.append(points_2d[i + num_points].astype(int))

    # Convert the lists to numpy arrays
    base_circle = np.array(base_circle)
    top_circle = np.array(top_circle)

    # Fill the polygons representing the cylinder's sides
    for i in range(num_points):
        polygon = np.array(
            [
                base_circle[i],
                base_circle[(i + 1) % num_points],
                top_circle[(i + 1) % num_points],
                top_circle[i],
            ]
        )
        cv2.fillPoly(canvas, [polygon], 255)

    # Fill the base and top circles
    cv2.drawContours(canvas, [base_circle.reshape((-1, 1, 2))], 0, 255, -1)
    cv2.drawContours(canvas, [top_circle.reshape((-1, 1, 2))], 0, 255, -1)

    return canvas


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union
    return iou


def cv2_imshow(image, winname="Image", flip_color=False):
    # Convert from BGR to RGB format
    # cv2.imshow(winname, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return
    if flip_color:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis("off")  # To hide axis values


def euler_to_rotation_matrix(alpha, beta, gamma):
    # Compute rotation matrix around x-axis
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    # Compute rotation matrix around y-axis
    R_y = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    # Compute rotation matrix around z-axis
    R_z = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    # Combine the individual rotation matrices
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return R


def rotation_matrix_to_euler_angles(R):
    alpha = np.arctan2(R[2, 1], R[2, 2])
    beta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    gamma = np.arctan2(R[1, 0], R[0, 0])

    return alpha, beta, gamma


def jaccard_index(mask1, mask2):
    union = np.logical_or(mask1, mask2).sum()
    assert union > 0
    return np.logical_and(mask1, mask2).sum() / union
