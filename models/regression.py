import torch.nn as nn
import torch.nn.functional as func
from features import GetFeatures

class Regression(nn.Module):
    def __init__(self, points):
        super(Regression, self).__init__()
        self.points = points

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 12)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        g_features, trans_feat = GetFeatures(self.points)

        x = self.drop(func.relu(self.bn1(self.fc1(x))))
        x = self.drop(func.relu(self.bn2(self.fc2(x))))
        x = self.drop(func.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x.view(-1, 4, 3), trans_feat