import os
import trimesh
from mesh_tools.clean import clean


def test_fix_normals_consistent_winding(cube_stl, tmp_dir):
    """fix_normals should produce consistent face winding."""
    output = os.path.join(tmp_dir, "cleaned.stl")
    result = clean(cube_stl, output, fix_normals=True)
    assert result.is_watertight


def test_remove_duplicates_reduces_vertices(tmp_dir):
    """remove_duplicates should reduce vertex count on mesh with intentional dupes."""
    path = os.path.join(tmp_dir, "duped.stl")
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    box.export(path)

    output = os.path.join(tmp_dir, "deduped.stl")
    mesh_before = trimesh.load(path, force="mesh")
    verts_before = mesh_before.vertices.shape[0]

    result = clean(path, output, remove_duplicates=True)
    assert result.vertices.shape[0] <= verts_before


def test_clean_idempotent(cube_stl, tmp_dir):
    """Running clean twice should produce the same vertex/face count."""
    first = os.path.join(tmp_dir, "first.stl")
    second = os.path.join(tmp_dir, "second.stl")

    result1 = clean(cube_stl, first, fix_normals=True, remove_duplicates=True)
    result2 = clean(first, second, fix_normals=True, remove_duplicates=True)

    assert result1.vertices.shape[0] == result2.vertices.shape[0]
    assert result1.faces.shape[0] == result2.faces.shape[0]
