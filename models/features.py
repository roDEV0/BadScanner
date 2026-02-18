import torch
import torch.nn as nn
import torch.nn.functional as func
from transform import Transform


class GetFeatures(nn.Module):
    def __init__(self, points):
        super(GetFeatures, self).__init__()
        self.points = points

        self.trans3 = Transform(3, points)
        self.trans64 = Transform(64, points)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(points, return_indices=True)

    def forward(self, x):
        batch = x.shape[0]

        trans_matrix = self.trans3(x)

        x = torch.bmm(x.transpose(2, 1), trans_matrix).transpose(2, 1)

        x = func.relu(self.bn1(self.conv1(x)))
        x = func.relu(self.bn1(self.conv2(x)))

        trans_feat = self.trans64(x)

        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

        x = func.relu(self.bn1(self.conv3(x)))
        x = func.relu(self.bn2(self.conv4(x)))
        x = func.relu(self.bn3(self.conv5(x)))

        g_features = self.pool(x).view(batch, -1)

        return g_features, trans_feat
