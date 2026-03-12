import open3d
import numpy


def cephalic_index(mesh, identify=False):
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

    labeled_points = numpy.array([x_top, x_bottom, y_top, y_bottom])

    if identify:
        x_line = numpy.linalg.norm(x_top - x_bottom)
        y_line = numpy.linalg.norm(y_top - y_bottom)

        cephalic_score = (x_line / y_line) * 100

        print(f"X_Line -> {x_line}, Y_Line -> {y_line}, Score -> {cephalic_score}")

        if cephalic_score > 90:
            return labeled_points, True

        return labeled_points, False

    return labeled_points


def cva_index(mesh, identify=False):
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

    if identify:
        trbl_line = numpy.linalg.norm(corners[0] - corners[3])
        tlbr_line = numpy.linalg.norm(corners[1] - corners[2])

        top_score = (trbl_line - tlbr_line) * 100
        cephalic_score = (
            top_score / trbl_line if trbl_line > tlbr_line else top_score / tlbr_line
        )

        if cephalic_score > 6.25:
            return corners, True

        return corners, False

    return corners
