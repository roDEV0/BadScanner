import torch.nn as nn
from models.pointnet2.abstract import SetAbstraction
import torch
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.abs1 = SetAbstraction(5, 512, 3, 64)
        self.abs2 = SetAbstraction(10, 256, 3+64, 128)
        self.abs3 = SetAbstraction(15, 128, 3+128, 256)

        self.fc1 = nn.Linear(259, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4 * 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.abs1(points)
        x = self.abs2(x)
        x = self.abs3(x)

        x = x.max(dim=0).values

        x = self.dropout(F.relu(self.bn1(self.fc1(x.unsqueeze(0)))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x.view(4, 3)