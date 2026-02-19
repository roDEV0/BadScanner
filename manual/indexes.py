import open3d
import numpy


def cephalic_index(mesh):
    vertices = numpy.asarray(mesh.vertices)

    z_threshold = numpy.percentile(vertices[:, 2], 95)
    top_head = numpy.array([vert for vert in vertices if vert[2] > z_threshold])

    top_head_mesh = open3d.geometry.PointCloud()
    top_head_mesh.points = open3d.utility.Vector3dVector(top_head)

    x_midline = (numpy.max(top_head[:, 0]) + numpy.min(top_head[:, 0])) / 2
    y_midline = (numpy.max(top_head[:, 1]) + numpy.min(top_head[:, 1])) / 2

    close_x = numpy.array(
        [vert for vert in top_head if (x_midline - 3) < vert[0] < (x_midline + 3)]
    )
    close_x = close_x[numpy.argsort(close_x[:, 1])]

    close_y = numpy.array(
        [vert for vert in top_head if (y_midline - 3) < vert[1] < (y_midline + 3)]
    )
    close_y = close_y[numpy.argsort(close_y[:, 0])]

    x_top = close_x[0]
    y_top = close_y[0]
    x_bottom = close_x[-1]
    y_bottom = close_y[-1]

    # pcd = mesh.sample_points_uniformly(number_of_points=1024)
    # sampled_points = numpy.asarray(pcd.points)

    labeled_points = numpy.array([x_top, x_bottom, y_top, y_bottom])
    # all_points = numpy.vstack([sampled_points, labeled_points])

    # final_pcd = open3d.geometry.PointCloud()
    # final_pcd.points = open3d.utility.Vector3dVector(all_points)

    print(labeled_points)


def cva_index(mesh):
    vertices = numpy.asarray(mesh.vertices)

    z_threshold = numpy.percentile(vertices[:, 2], 95)
    top_head = numpy.array([vert for vert in vertices if vert[2] > z_threshold])

    top_head_mesh = open3d.geometry.PointCloud()
    top_head_mesh.points = open3d.utility.Vector3dVector(top_head)

    center3d = numpy.array(
        [
            numpy.mean(top_head[:, 0]),
            numpy.mean(top_head[:, 1]),
            numpy.mean(top_head[:, 2]),
        ]
    )

    other_angles = numpy.arctan2(
        top_head[:, 0] - center3d[0], top_head[:, 1] - center3d[1]
    )

    targets = [numpy.pi / 4, 3 * numpy.pi / 4, -numpy.pi / 4, -3 * numpy.pi / 4]

    corners = []
    for comparison in targets:
        angle_diff = abs(other_angles - comparison)

        tolerance = angle_diff < numpy.radians(7)
        remaining = top_head[tolerance]

        distances = numpy.linalg.norm(remaining[:, :2] - center3d[:2], axis=1)
        corners.append(remaining[numpy.argmax(distances)])

    corners = numpy.array(corners)

    pcd = mesh.sample_points_uniformly(number_of_points=1024)
    sampled_points = numpy.asarray(pcd.points)

    all_points = numpy.vstack([sampled_points, corners, center3d.reshape(1, 3)])

    final_pcd = open3d.geometry.PointCloud()
    final_pcd.points = open3d.utility.Vector3dVector(all_points)
