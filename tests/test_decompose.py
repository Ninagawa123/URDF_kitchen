import os
import pytest
from mesh_tools.decompose import decompose


def test_cube_decomposes_to_one_hull(cube_stl, tmp_dir):
    """Cube is already convex, should decompose into exactly 1 hull."""
    output = os.path.join(tmp_dir, "cube_collision.stl")
    num_hulls, info = decompose(cube_stl, output)
    assert num_hulls == 1
    assert os.path.exists(output)


def test_l_shape_decomposes_to_multiple_hulls(l_shape_stl, tmp_dir):
    """L-shape is concave, should decompose into 2+ hulls."""
    output = os.path.join(tmp_dir, "l_collision.stl")
    num_hulls, info = decompose(l_shape_stl, output)
    assert num_hulls >= 2
    assert os.path.exists(output)


def test_no_merge_produces_individual_files(l_shape_stl, tmp_dir):
    """merge=False should produce individual _hull_NNN files."""
    output = os.path.join(tmp_dir, "l_collision.stl")
    num_hulls, info = decompose(l_shape_stl, output, merge=False)
    for i in range(num_hulls):
        hull_path = os.path.join(tmp_dir, f"l_collision_hull_{i:03d}.stl")
        assert os.path.exists(hull_path), f"Missing hull file: {hull_path}"


def test_missing_input_raises_error(tmp_dir):
    """Decomposing a non-existent file should raise an error."""
    with pytest.raises(Exception):
        decompose("/nonexistent/path.stl", os.path.join(tmp_dir, "out.stl"))


def test_output_file_exists_after_decompose(cube_stl, tmp_dir):
    """Output file should exist after a successful decompose."""
    output = os.path.join(tmp_dir, "output.stl")
    num_hulls, info = decompose(cube_stl, output)
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
