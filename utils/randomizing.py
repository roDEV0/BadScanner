import numpy
import open3d


def random_rotation(pcd: open3d.geometry.PointCloud):
    rotation = pcd.get_rotation_matrix_from_xyz(numpy.random.randn(3))
    pcd.rotate(rotation, center=pcd.get_center())

    return pcd


def random_scale(pcd: open3d.geometry.PointCloud):
    pcd.scale(scale=abs(numpy.random.randn()), center=pcd.get_center())

    return pcd


def regional_dropout(pcd, num_regions=3, region_radius_ratio=0.1):
    points = open3d.geometry.PointCloud()
    points.points = open3d.utility.Vector3dVector(pcd)

    # Compute a relative radius based on the cloud's extent
    bbox = points.get_axis_aligned_bounding_box()
    extent = numpy.linalg.norm(bbox.get_extent())
    region_radius = region_radius_ratio * extent

    mask = numpy.ones(len(points.points), dtype=bool)

    for _ in range(num_regions):
        center = points.points[numpy.random.randint(len(points.points))]
        dists = numpy.linalg.norm(points.points - center, axis=1)
        mask &= dists > region_radius

    result = open3d.geometry.PointCloud()
    result_points = numpy.asarray(points.points)[mask]
    result.points = open3d.utility.Vector3dVector(result_points)
    return result


def sample_fixed_points(pcd, num_points=1024):
    points = numpy.asarray(pcd.points)
    n = len(points)

    if n >= num_points:
        idx = numpy.random.choice(n, num_points, replace=False)
    else:
        idx = numpy.random.choice(n, num_points, replace=True)

    result = open3d.geometry.PointCloud()
    result.points = open3d.utility.Vector3dVector(points[idx])
    return result

def generate_random(pcd):
    randomized = regional_dropout(pcd)

    randomized = random_scale(randomized)
    randomized = random_rotation(randomized)
    randomized = sample_fixed_points(randomized)

    return numpy.asarray(randomized.points, dtype=numpy.float32)