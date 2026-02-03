from dataloaders.fromobj import ObjToPointCloud

dataset = ObjToPointCloud('./objects', 1024)

print(dataset.__getitem__(0))