from dataloaders.fromobj import ObjToPointCloud

dataset = ObjToPointCloud('./objects/models', 1024)

print(dataset.load_all(save=True))