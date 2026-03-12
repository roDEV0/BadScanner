import torch
import open3d as o3d
import numpy as np
import scipy.spatial as sp
import numpy as np


# Needs to be N x 3 size
def farthest_point_sample(points: torch.Tensor, num_cen: int) -> torch.Tensor:
    print(points.shape)

    points_vector = np.asarray(points.cpu(), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vector)

    pcd = pcd.farthest_point_sample(num_cen)

    points_tensor = torch.from_numpy(np.asarray(pcd.points)).float()

    # Returns N' x 3
    return points_tensor


# Needs to be N' x 3 and N x 3 size
def ball_query(
    centroid: torch.Tensor, points: torch.Tensor, radius: float, samples: int
) -> torch.Tensor:
    neighbors = o3d_ml.ops.ball_query(points, centroid, radius, samples)

    # Returns K x 3
    return torch.tensor(neighbors)
