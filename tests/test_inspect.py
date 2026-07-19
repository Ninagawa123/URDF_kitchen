import json
import trimesh
from mesh_tools.inspect_mesh import inspect


def test_cube_watertight_and_convex(cube_stl):
    """Cube should report watertight=True and convex=True."""
    stats = inspect(cube_stl)
    assert stats['watertight'] is True
    assert stats['convex'] is True


def test_json_output_valid_and_has_keys(cube_stl):
    """inspect output should be serializable to valid JSON with expected keys."""
    stats = inspect(cube_stl)
    json_str = json.dumps(stats)
    parsed = json.loads(json_str)
    expected_keys = {'file', 'vertices', 'faces', 'watertight', 'volume',
                     'bounds', 'center_mass', 'convex', 'file_size'}
    assert expected_keys == set(parsed.keys())


def test_correct_vertex_and_face_counts(cube_stl):
    """inspect should report the same vertex/face counts as trimesh."""
    mesh = trimesh.load(cube_stl, force="mesh")
    stats = inspect(cube_stl)
    assert stats['vertices'] == mesh.vertices.shape[0]
    assert stats['faces'] == mesh.faces.shape[0]
