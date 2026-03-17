import torch
import torch_fpsample

# Needs to be B x N x 3 size
def farthest_point_sample(points: torch.Tensor, n_centroids: int) -> torch.Tensor:
    if points.shape[-1] != 3:
        points = points.permute(0, 2, 1)

    batch_centroids, _ = torch_fpsample.sample(points.cpu(), n_centroids)
    return batch_centroids.to(points.device)

# Needs to be B x N' x 3 and B x N x 3 and B x N x C size
def ball_query(centroids: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor, radius: float, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    if xyz.shape[-1] != 3:
        xyz = xyz.permute(0, 2, 1)

    if features.shape[1] != xyz.shape[1]:
        features = features.permute(0, 2, 1)

    B, Np, _ = centroids.shape
    N = xyz.shape[1]
    C = features.shape[2]

    dists = torch.cdist(centroids, xyz)  # B x N' x N

    sorted_dists, sorted_idx = torch.sort(dists, dim=-1)
    neighbor_idx = sorted_idx[:, :, :samples]
    neighbor_dists = sorted_dists[:, :, :samples]  # B x N' x K

    nearest = sorted_idx[:, :, 0].unsqueeze(-1)  # B x N' x 1
    neighbor_idx = torch.where(
        neighbor_dists < radius,
        neighbor_idx,
        nearest.expand(B, Np, samples),
    )

    # Gather xyz neighbors: B x N' x K x 3
    idx_xyz = neighbor_idx.unsqueeze(-1).expand(B, Np, samples, 3)
    xyz_exp = xyz.unsqueeze(1).expand(B, Np, N, 3)
    xyz_neighbors = torch.gather(xyz_exp, 2, idx_xyz)

    # Gather feature neighbors: B x N' x K x C
    idx_feat = neighbor_idx.unsqueeze(-1).expand(B, Np, samples, C)
    feat_exp = features.unsqueeze(1).expand(B, Np, N, C)
    feature_neighbors = torch.gather(feat_exp, 2, idx_feat)

    return xyz_neighbors, feature_neighbors