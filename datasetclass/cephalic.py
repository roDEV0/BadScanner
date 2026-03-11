import torch
from torch.utils.data import Dataset
from utils.randomizing import generate_random
from pathlib import Path
import numpy

class HeadScanDataset(Dataset):
    def __init__(self, directory, randomize=False):
        self.directory = Path(directory)
        self.randomize = randomize
        self.paths = [file for file in self.directory.iterdir() if file.suffix == ".npz"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        data = numpy.load(self.paths[item])

        cloud = data["cloud"].astype(numpy.float32)
        truths = data["truths"].astype(numpy.float32)
        identity = data["plageo"].astype(numpy.bool)

        if self.randomize:
            cloud = generate_random(cloud)

        cloud = torch.tensor(cloud).transpose(0, 1)
        truths = torch.tensor(truths)

        return cloud, truths, identity
