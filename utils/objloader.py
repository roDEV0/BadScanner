import open3d

def load_mesh(path):
    return open3d.io.read_triangle_mesh(path)
