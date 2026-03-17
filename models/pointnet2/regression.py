import torch.nn as nn
from models.pointnet2.abstract import SetAbstraction
import torch
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.abs1 = SetAbstraction(50, 50, 512, 3, 48)
        self.abs2 = SetAbstraction(100, 50, 256, 48, 128)
        self.abs3 = SetAbstraction(150, 50, 128, 128, 256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4 * 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        features = points

        points, features = self.abs1(points, features)
        print("Layer 1 complete")
        points, features = self.abs2(points, features)
        print("Layer 2 complete")
        points, features = self.abs3(points, features)

        features = features.max(dim=1).values

        x = self.dropout(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x.view(batch_size, 4, 3)