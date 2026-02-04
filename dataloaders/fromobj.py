import os
import torch
import pytorch3d.ops as ops
from torch.utils.data import Dataset
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
import warnings


class ObjToPointCloud(Dataset):
    def __init__(self, dir, points, preload=False):
        self.dir = dir
        self.points = points
        self.paths = sorted([os.path.join(dir, path) for path in os.listdir(dir) if path.endswith('.obj')])

        self.preloaded_data = None
        if preload:
            self.preloaded_data = self.load_all()

    def __len__(self):
        return len(self.paths)

    # Converts a specific item in the database to a PointCloud
    def load_item(self, index, save=False):
        if self.preloaded_data is not None:
            return self.preloaded_data[index]

        path = self.paths[index]
        warnings.filterwarnings('ignore')
        mesh = IO().load_mesh(path)
        pcl = ops.sample_points_from_meshes(mesh, self.points)

        if save:
            IO().save_pointcloud(Pointclouds(points=[pcl.squeeze()]), path.replace('.obj', '.ply'))

        return pcl

    # Converts all objects in a directory to a PointCloud
    def load_all(self, save=False):
        point_clouds = []
        warnings.filterwarnings('ignore')

        for path in self.paths:
            mesh = IO().load_mesh(path)
            pcl = ops.sample_points_from_meshes(mesh, self.points)

            if save:
                IO().save_pointcloud(Pointclouds(points=[pcl.squeeze()]), path.replace('.obj', '.ply'))
            point_clouds.append(pcl)

        return torch.cat(point_clouds, dim=0)