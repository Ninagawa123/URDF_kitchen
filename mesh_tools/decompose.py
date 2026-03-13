import os
import sys
import trimesh
import coacd

from mesh_tools.utils import ensure_output_dir


def decompose(input_path, output_path, threshold=0.05, max_hulls=-1,
              preprocess="auto", resolution=2000, mcts_nodes=20,
              mcts_iterations=150, merge=True):
    """Run CoACD convex decomposition on a mesh.

    Returns (num_hulls, result_info) where result_info is a dict with stats.
    """
    mesh = trimesh.load(input_path, force="mesh")
    input_faces = int(mesh.faces.shape[0])
    input_verts = int(mesh.vertices.shape[0])

    # Warn if bounds suggest wrong units
    extents = mesh.bounding_box.extents
    if any(e > 10.0 for e in extents):
        print(f"WARNING: Mesh extents {extents} exceed 10m. Are your units meters?", file=sys.stderr)

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode=preprocess,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
    )

    ensure_output_dir(output_path)

    output_faces = 0
    output_verts = 0
    hull_meshes = []

    if merge:
        hull_meshes = [trimesh.Trimesh(vertices=vs, faces=fs) for vs, fs in parts]
        combined = trimesh.util.concatenate(hull_meshes)
        combined.export(output_path)
        output_faces = int(combined.faces.shape[0])
        output_verts = int(combined.vertices.shape[0])
    else:
        base, ext = os.path.splitext(output_path)
        for i, (vs, fs) in enumerate(parts):
            hull = trimesh.Trimesh(vertices=vs, faces=fs)
            hull_meshes.append(hull)
            hull_path = f"{base}_hull_{i:03d}{ext}"
            hull.export(hull_path)
            output_faces += int(fs.shape[0])
            output_verts += int(vs.shape[0])

    num_hulls = len(parts)
    result_info = {
        'num_hulls': num_hulls,
        'input_faces': input_faces,
        'input_verts': input_verts,
        'output_faces': output_faces,
        'output_verts': output_verts,
        'hull_meshes': hull_meshes,
    }

    return num_hulls, result_info
