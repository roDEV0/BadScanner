import torch
import open3d as o3d
import numpy as np
import os


# Needs to be B x N x 3 size
def farthest_point_sample(points: torch.Tensor, n_centroids: int) -> torch.Tensor:
    with torch.no_grad():
        if points.shape[-1] != 3:
            points = points.permute(0, 2, 1)

        batch_size, n_points, _ = points.shape
        device = points.device

        selected = torch.zeros(batch_size, n_centroids, dtype=torch.long, device=device)
        selected[:, 0] = torch.randint(0, n_points, (batch_size,), device=device)

        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

        for i in range(1, n_centroids):
            last_points = points[batch_indices, selected[:, i-1]].unsqueeze(1)  # (B,1,3)
            distances = torch.cdist(last_points, points)  # (B,1,N)
            selected[:, i] = torch.argmax(distances.squeeze(1), dim=1)

        batch_centroids = torch.gather(points, 1, selected.unsqueeze(-1).expand(-1, -1, 3))

        return batch_centroids

# Needs to be B x N' x 3 and B x N x 3 and B x N x C size
def ball_query(centroids: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor, radius: float, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    if xyz.shape[-1] != 3:
        xyz = xyz.permute(0, 2, 1)

    if features.shape[1] != xyz.shape[1]:
        features = features.permute(0, 2, 1)

    batch_size, n_centroids, _ = centroids.shape
    n_points = xyz.shape[1]
    n_features = features.shape[2]

    dists = torch.cdist(centroids, xyz)

    within_radius = dists < radius

    _, sorted_idx = torch.sort(within_radius.float(), dim=-1, descending=True)
    neighbor_idx = sorted_idx[:, :, :samples]

    has_neighbor = within_radius.any(dim=-1)
    nearest = torch.argmin(dists, dim=-1)
    fallback = nearest.unsqueeze(-1).expand(batch_size, n_centroids, samples)

    neighbor_idx = torch.where(has_neighbor.unsqueeze(-1), neighbor_idx, fallback)

    idx_xyz = neighbor_idx.unsqueeze(-1).expand(batch_size, n_centroids, samples, 3)
    xyz_exp = xyz.unsqueeze(1).expand(batch_size, n_centroids, n_points, 3)
    xyz_neighbors = torch.gather(xyz_exp, 2, idx_xyz)

    idx_feat = neighbor_idx.unsqueeze(-1).expand(batch_size, n_centroids, samples, n_features)
    feat_exp = features.unsqueeze(1).expand(batch_size, n_centroids, n_points, n_features)
    feature_neighbors = torch.gather(feat_exp, 2, idx_feat)

    return xyz_neighbors, feature_neighbors