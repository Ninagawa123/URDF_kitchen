import os
import trimesh

from mesh_tools.utils import format_filesize


def inspect(input_path):
    """Analyze a mesh and return statistics dict."""
    mesh = trimesh.load(input_path, force="mesh")
    extents = mesh.bounding_box.extents
    center_mass = mesh.center_mass.tolist() if mesh.is_watertight else [0, 0, 0]

    stats = {
        'file': os.path.basename(input_path),
        'vertices': int(mesh.vertices.shape[0]),
        'faces': int(mesh.faces.shape[0]),
        'watertight': bool(mesh.is_watertight),
        'volume': float(mesh.volume) if mesh.is_watertight else None,
        'bounds': [float(f"{e:.3f}") for e in extents],
        'center_mass': [float(f"{c:.3f}") for c in center_mass],
        'convex': bool(mesh.is_convex),
        'file_size': os.path.getsize(input_path),
    }
    return stats
