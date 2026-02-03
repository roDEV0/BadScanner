import os
import pytorch3d.ops as ops
from torch.utils.data import Dataset
from pytorch3d.io import IO
import warnings

class ObjToPointCloud(Dataset):
    def __init__(self, dir, points):
        self.dir = dir
        self.points = points

        self.paths = sorted([os.path.join(dir, path) for path in os.listdir(dir) if path.endswith('.obj')])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        warnings.filterwarnings('ignore')

        mesh = IO().load_mesh(path)
        pcl = ops.sample_points_from_meshes(mesh, self.points)
        return pcl