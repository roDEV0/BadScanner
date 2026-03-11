import numpy
import open3d
from datasetclass.cephalic import HeadScanDataset
from pathlib import Path
import torch
from models.cephalic.cpnet import CRegression
from models.cvai.cvpnet import CVRegression
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Path(__file__).resolve().parent.parent
NPZ_DIR = root / "dataset/cephalic"
MODELS_DIR = root / "checkpoint/checkpoint.pth"
POINT_COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

dataset = HeadScanDataset(NPZ_DIR)
checkpoint = torch.load(MODELS_DIR)

target_mean = checkpoint['target_mean'].to(device)
target_std  = checkpoint['target_std'].to(device)

model = CRegression(1028).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

iterator = iter(loader)

head, truths, plageo = next(iterator)

head_np = numpy.asarray(head.squeeze(0).transpose(0, 1), dtype=numpy.float64)

head = head.to(device)

with torch.no_grad():
    output = model(head) * target_std + target_mean
    output_cpu = output.cpu()

output_cpu = numpy.asarray(output_cpu.squeeze(0), dtype=numpy.float64)

pcd_head = open3d.geometry.PointCloud()
pcd_head.points = open3d.utility.Vector3dVector(head_np)
pcd_head.paint_uniform_color([0.5, 0.5, 0.5])

pcd_points = open3d.geometry.PointCloud()
pcd_points.points = open3d.utility.Vector3dVector(output_cpu)

pcd_points.paint_uniform_color([1.0, 0.0, 0.0])

lines = [[0, 1], [2, 3]]
line_drawing = open3d.geometry.LineSet()

line_drawing.points = open3d.utility.Vector3dVector(output_cpu)
line_drawing.lines = open3d.utility.Vector2iVector(lines)

length_one = numpy.linalg.norm(output_cpu[0] - output_cpu[1])
length_two = numpy.linalg.norm(output_cpu[2] - output_cpu[3])

# plageo = (length_one - length_two) * 100
# plageo = plageo/length_one if length_one > length_two else plageo/length_two
# plageo = plageo * 1.5
#
# if plageo < 3.5:
#     plageo_check = "None"
# elif 3.5 < plageo < 6.25:
#     plageo_check = "Minimal"
# elif 6.25 < plageo < 8.75:
#     plageo_check = "Moderate"
# elif 8.75 < plageo < 11:
#     plageo_check = "High"
# elif plageo > 11:
#     plageo_check = "Severe"

plageo_check = (length_one/length_two) * 100

print(f"Top-Down Length: {length_two}")
print(f"Left-Right Length: {length_one}")
print(f"Plageocephaly?: {plageo_check > 90} - Score: {plageo_check}")
print(f"Truth: {plageo}")

open3d.visualization.draw_geometries([pcd_head, pcd_points, line_drawing])