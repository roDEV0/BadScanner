import copy
import numpy
from pathlib import Path
from manual.objloader import load_mesh
from utils.orientation import random_rotation, random_scale, regional_dropout


def generate_augmented_pair(mesh_path, num_points=1028):
    import_mesh = load_mesh(mesh_path)
    original = import_mesh.sample_points_uniformly(number_of_points=num_points)
    augmented = copy.deepcopy(original)

    augmented = random_scale(augmented)
    augmented = random_rotation(augmented)
    augmented = regional_dropout(augmented)

    return (
        numpy.asarray(original.points),
        numpy.asarray(augmented.points),
    )


def save_pair(original_points, augmented_points, output_path):
    numpy.savez(
        output_path,
        original=original_points,
        augmented=augmented_points,
    )


training_dir = Path("/objects")
output_dir = Path("/dataset")

for model in training_dir.iterdir():
    mesh_path = model
    print(mesh_path)
    output_dir.mkdir(exist_ok=True)

    num_augmentations = 10
    for i in range(num_augmentations):
        original_pts, augmented_pts = generate_augmented_pair(mesh_path)
        save_pair(original_pts, augmented_pts, output_dir / f"{mesh_path.stem}{i:04d}")
