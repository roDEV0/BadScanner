import models.pointnet2.pnet_utils as pnet_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


# Requires B x N x 3 and B x N x C
class SampleAndGroup(nn.Module):
    def __init__(self, samples: int, radius: float, max_neighbors: int):
        super().__init__()
        self.samples = samples
        self.radius = radius
        self.max_neighbors = max_neighbors

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # xyz is the B x N x 3 tensor
        with torch.no_grad():
            batch_centroids = pnet_utils.farthest_point_sample(xyz, self.samples)
        batch_xyz, batch_features = pnet_utils.ball_query(batch_centroids, xyz, features, self.radius, self.max_neighbors)

        # Returns B x N' x 3 and B x N' x K x 3 and B x N' x K x C
        return batch_centroids, batch_xyz, batch_features


# Requires B x N' x K x 3 and B x N' x K x C and B x N' x 3
class PointNet(nn.Module):
    def __init__(self, dims: int, features: int):
        super().__init__()

        self.dims = dims
        self.features = features

        self.conv1 = nn.Conv1d(dims, dims * 2, 1)
        self.conv2 = nn.Conv1d(dims * 2, dims * 4, 1)
        self.conv3 = nn.Conv1d(dims * 4, dims * 8, 1)
        self.conv4 = nn.Conv1d(dims * 8, features, 1)

        self.bn1 = nn.BatchNorm1d(dims * 2)
        self.bn2 = nn.BatchNorm1d(dims * 4)
        self.bn3 = nn.BatchNorm1d(dims * 8)
        self.bn4 = nn.BatchNorm1d(features)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Local frame: subtract centroid from coords only -> B x N' x K x 3/C
        relative_coords = xyz - centroids.unsqueeze(2)

        # Keep input channels consistent with constructor: [relative_coords(3) + features(dims)].
        x = torch.cat([relative_coords, features], dim=-1)

        B, Np, K, C_in = x.shape

        x = x.permute(0, 1, 3, 2).reshape(B * Np, C_in, K)


        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        x = F.max_pool1d(x, x.size(-1))  # B*N' x C' x 1
        x = x.squeeze(-1)  # B*N' x C'

        x = x.reshape(B, Np, x.shape[1])

        return x, centroids


# Requires B x N x 3 and B x N x C
class SetAbstraction(nn.Module):
    def __init__(self, radius: float, max_neighbors: int, n_centroids: int, dims: int, features: int):
        super().__init__()
        self.radius = radius
        self.sample_and_group = SampleAndGroup(n_centroids, radius, max_neighbors)
        # PointNet receives concatenated [relative_coords(3) + features(dims)]
        self.pointnet = PointNet(3 + dims, features)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_centroids, new_xyz, new_features = self.sample_and_group(xyz, features)
        new_features, new_centroids = self.pointnet(new_xyz, new_features, new_centroids)

        # Returns B x N' x 3 and B x N' x C
        return new_centroids, new_features
