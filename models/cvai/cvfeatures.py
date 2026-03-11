import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cephalic.ctnet import CTnet


class CVGetFeatures(nn.Module):
    def __init__(self, num_points=1028, num_global_feats=8):
        super(CVGetFeatures, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats

        self.tnet1 = CTnet(dim=3, num_points=num_points)
        self.tnet2 = CTnet(dim=64, num_points=num_points)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        self.max_pool = nn.AdaptiveMaxPool1d(1, return_indices=True)

    def forward(self, x):

        bs = x.shape[0]

        A_input = self.tnet1(x)

        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        A_feat = self.tnet2(x)

        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)

        return global_features