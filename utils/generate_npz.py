import numpy
from pathlib import Path
from utils.objloader import load_mesh
from manual.indexes import cephalic_index, cva_index

def determine_points(mesh_path, cephalic: bool):
    import_mesh = load_mesh(mesh_path)
    if cephalic:
        return cephalic_index(import_mesh, True)
    return cva_index(import_mesh, True)

def save_trio(original_points, ground_truths, output_path, plageo_truth=None):
    numpy.savez(
        output_path,
        cloud = original_points,
        truths = ground_truths,
        plageo = plageo_truth
    )

def create_npz(cephalic=True):
    root = Path(__file__).resolve().parent.parent
    obj_dir = root / "objects"
    out_dir = root / "dataset"

    training_dir = Path(obj_dir)
    output_dir = Path(out_dir)

    for model in training_dir.iterdir():
        path = model
        output_dir.mkdir(exist_ok=True)

        path_mesh = load_mesh(path)
        if cephalic:
            truths, identity = determine_points(path, True)
            cloud = path_mesh.sample_points_uniformly(number_of_points=1028)
            save_trio(numpy.array(cloud.points), truths, output_dir / f"cephalic/{path.stem}", identity)
        else:
            truths, identity = determine_points(path, False)
            cloud = path_mesh.sample_points_uniformly(number_of_points=1028)
            save_trio(numpy.array(cloud.points), truths, output_dir / f"cvai/{path.stem}", identity)

        print(f"Finished generating {path}")

create_npz(True)