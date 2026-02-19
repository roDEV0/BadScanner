import open3d
import numpy

# This is a new algorithm I may try to use, but at the moment it is slower than the other algorithms
# Instead of doing angle comparisons or means, this uses ray casting

# TODO: Optimize this

def ca(mesh: open3d.geometry.TriangleMesh):
    t_mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)

    verts = t_mesh.vertex.positions.numpy()

    threshold = numpy.percentile(verts[:, 2], 95)
    top_verts = verts[verts[:, 2] >= threshold]

    center = numpy.mean(top_verts, axis=0)
    x_center, y_center, z_center = center

    rays = open3d.core.Tensor(
        [
            [x_center, y_center, z_center, 1, 0, 0],
            [x_center, y_center, z_center, -1, 0, 0],
            [x_center, y_center, z_center, 0, 1, 0],
            [x_center, y_center, z_center, 0, -1, 0],
        ],
        dtype=open3d.core.Dtype.Float32,
    )

    ray_scene = open3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(t_mesh)

    response = ray_scene.cast_rays(rays)

    multipliers = response["t_hit"].numpy()

    directions = numpy.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=numpy.float32
    )
    hit_points = center + multipliers[:, None] * directions

    for hit_point in hit_points:
        print(f"Exact hit point: {hit_point}")
