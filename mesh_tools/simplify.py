import trimesh

from mesh_tools.utils import ensure_output_dir


def simplify(input_path, output_path, target_faces=None, ratio=None):
    """Simplify a mesh via quadric decimation. Returns the simplified trimesh."""
    mesh = trimesh.load(input_path, force="mesh")
    original_faces = mesh.faces.shape[0]

    if ratio is not None:
        target_faces = int(original_faces * ratio)

    simplified = mesh.simplify_quadric_decimation(target_faces)
    ensure_output_dir(output_path)
    simplified.export(output_path)

    return simplified
