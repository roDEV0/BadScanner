import numpy
import open3d
import copy

from manual.objloader import load_mesh


def random_rotation(pcd: open3d.geometry.PointCloud):
    rotation = pcd.get_rotation_matrix_from_xyz(numpy.random.randn(3))
    pcd.rotate(rotation, center=pcd.get_center())

    return pcd


def random_scale(pcd: open3d.geometry.PointCloud):
    pcd.scale(scale=abs(numpy.random.randn()), center=pcd.get_center())

    return pcd


def regional_dropout(pcd, num_regions=3, region_radius_ratio=0.1):
    points = numpy.asarray(pcd.points)

    # Compute a relative radius based on the cloud's extent
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = numpy.linalg.norm(bbox.get_extent())
    region_radius = region_radius_ratio * extent

    mask = numpy.ones(len(points), dtype=bool)

    for _ in range(num_regions):
        center = points[numpy.random.randint(len(points))]
        dists = numpy.linalg.norm(points - center, axis=1)
        mask &= dists > region_radius

    result = open3d.geometry.PointCloud()
    result.points = open3d.utility.Vector3dVector(points[mask])
    return result
