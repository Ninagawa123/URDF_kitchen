import trimesh

from mesh_tools.utils import ensure_output_dir


def clean(input_path, output_path, fix_normals=False, remove_duplicates=False,
          fill_holes=False, merge_tolerance=0.0001):
    """Clean/repair a mesh and export to output_path. Returns the cleaned trimesh."""
    mesh = trimesh.load(input_path, force="mesh")

    if fix_normals:
        mesh.fix_normals()

    if remove_duplicates:
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())

    if fill_holes:
        trimesh.repair.fill_holes(mesh)

    ensure_output_dir(output_path)
    mesh.export(output_path)
    return mesh
