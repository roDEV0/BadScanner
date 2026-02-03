import torch
import torch.nn as nn
import torch.nn.functional as func

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else torch.device('cpu')
print(f'Using {device} device')

class Transform(nn.Module):
    def __init__(self, dim, points):
        super(Transform, self).__init__()
        self.dim = dim

        # Starting at 3 because X, Y, Z
        self.conv1 = nn.Conv1d(dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Extract features from each point
        self.pool = nn.MaxPool1d(points)

    def forward(self, x):
        batch = x.shape[0]

        x = func.relu(self.bn1(self.conv1(x)))
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).view(batch, -1)

        x = func.relu(self.bn2(self.fc1(x)))
        x = func.relu(self.bn1(self.fc2(x)))
        x = self.fc3(x)

        # Format for point
        identity = torch.eye(self.dim, requires_grad=True).repeat(batch, 1, 1)

        return x.view(-1, self.dim, self.dim) + identity