import models.pointnet2.pnet_utils as pnet_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


# Requires B x N x (3 + C)
class SampleAndGroup(nn.Module):
    def __init__(self, samples: int, radius: float):
        super().__init__()
        self.samples = samples
        self.radius = radius

    def forward(
        self, points: torch.Tensor, d: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = points[:, :, :d]  # B x N x 3
        centroids = torch.stack([pnet_utils.farthest_point_sample(item, self.samples) for item in coords])  # N' x 3
        neighbor_tensor = torch.stack([torch.stack(
            [
                pnet_utils.ball_query(center, model_points, self.radius, self.samples)
                for center in centroids
            ],
            dim=0,
        ) for model_points in coords])  # B x N' x K x (3 + C)

        # Returns B x N' x K x (3 + C) and B x N' x 3
        return neighbor_tensor, centroids


# Requires B x N' x K x (3 + C) and B x N' x 3
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

    def forward(
        self, points: torch.Tensor, centroids: torch.Tensor, d: int
    ) -> torch.Tensor:
        coords = points[:, :, :, :d]  # B x N' x K x d
        feats = points[:, :, :, d:]  # B x N' x K x C

        # Local frame: subtract centroid from coords only
        local_coords = coords - centroids.unsqueeze(2)  # B x N' x K x d
        x = torch.cat([local_coords, feats], dim=-1)  # B x N' x K x (d+C)

        x = x.permute(0, 1, 3, 2)

        x = x.permute(0, 1, 3, 2)  # B x N' x (3 + C) x K

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        x = F.max_pool1d(x, x.size(-1))  # N' x C' x 1
        x = x.squeeze(-1)  # N' x C'

        return x


# Requires N x (3 + C)
class SetAbstraction(nn.Module):
    def __init__(self, radius: float, centroids: int, dims: int, features: int):
        super().__init__()
        self.d = 3
        self.radius = radius
        self.sample_and_group = SampleAndGroup(centroids, radius)
        self.pointnet = PointNet(dims, features)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        neighbor_tensor, centroids = self.sample_and_group(points, self.d)
        features = self.pointnet(neighbor_tensor, centroids, self.d)

        # Returns N' x (3 + C')
        return torch.cat([centroids, features], dim=1)
